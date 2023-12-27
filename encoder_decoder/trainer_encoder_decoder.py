from imports.common_imports import *

from utils.utils import *
from utils.ema import EMA

from encoder_decoder.encoder_decoder import VQFIGAN
from encoder_decoder.discriminator import NLayerDiscriminator

class TrainerEncoderDecoder(pl.LightningModule):
    def __init__(self, test_dataloader, use_ema=False):
        super(TrainerEncoderDecoder, self).__init__()

        self.mean, self.sd = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        self.vqfigan = VQFIGAN()
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                                 n_layers=3,
                                                 use_actnorm=False,
                                                 ndf=64
                                                 )
        self.val_gan_factor = 0.8

        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(0.995)
            self.ema_vqfigan = copy.deepcopy(self.vqfigan).eval().requires_grad_(False)

        self.lpips_loss = LPIPS()
        self.charbonnier_loss = lambda x, y: torch.mean(torch.sqrt((x - y)**2 + 1e-6))

        self.train_metrics = torchmetrics.MetricCollection({
            "train_lpips": LPIPS(),
            "train_psnr": PSNR(),
            "train_ssim": SSIM()
        })
        self.val_metrics = torchmetrics.MetricCollection({
            "val_lpips": LPIPS(),
            "val_psnr": PSNR(),
            "val_ssim": SSIM()
        })

        self.test_dataloader = test_dataloader

        self.automatic_optimization = False

    def forward(self, target, I0, I1, in_trainig=False):
        if self.use_ema and not in_trainig:
            decoded_images, vq_loss = self.ema_vqfigan(target, I0, I1)
        else:
            decoded_images, vq_loss = self.vqfigan(target, I0, I1)
        return decoded_images, vq_loss 

    def perc_loss(self, x, x_hat):
        x_hat = torch.clamp(x_hat, -1, 1)
        percep_loss = self.lpips_loss(x, x_hat)
        pix2pix_loss = self.charbonnier_loss(x, x_hat)
        return pix2pix_loss + percep_loss

    def gan_adaptive_weight(self, perc_loss, gan_loss):
        last_layer = self.vqfigan.decoder.conv_out.weight
        perc_grads = torch.autograd.grad(perc_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(perc_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.val_gan_factor
        return d_weight

    def training_step(self, batch, batch_idx):
        I0, target, I1 = batch.transpose(0,1)
        
        optimizer_g, optimizer_d = self.optimizers()

        decoded_images, vq_loss = self.forward(target, I0, I1, in_trainig = True)
        #lossG = self.perc_loss(target, decoded_images) + vq_loss

        ## Gener ## 
        optimizer_g.zero_grad()
        reconstruction_loss = self.perc_loss(target, decoded_images)
        gan_loss = -torch.mean(self.discriminator(decoded_images))
        gan_ada_weight = self.gan_adaptive_weight(reconstruction_loss, gan_loss) 
        lossG = reconstruction_loss + vq_loss + self.val_gan_factor*gan_ada_weight*gan_loss
        self.manual_backward(lossG)
        optimizer_g.step()

        ## Discr ##    
        optimizer_d.zero_grad()
        d_loss_real = torch.mean(F.relu(1. - self.discriminator(target)))
        d_loss_fake = torch.mean(F.relu(1. + self.discriminator(decoded_images.detach())))
        lossD = 0.5 * (d_loss_real + d_loss_fake)
        self.manual_backward(lossD)
        optimizer_d.step()

        if self.use_ema:
            self.ema.step_ema(self.ema_vqfigan, self.vqfigan)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_lossG", lossG, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_lossD", lossD, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        with torch.inference_mode():
            mets = self.train_metrics(target, decoded_images.clamp(-1, 1))
            for k,v in mets.items():
                self.log(k, v, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
        # return lossG

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        I0, target, I1 = batch.transpose(0,1)
        
        decoded_images, vq_loss = self.forward(target, I0, I1, in_trainig = False)
        lossG = self.perc_loss(target, decoded_images) + vq_loss

        disc_real = self.discriminator(target)
        disc_fake = self.discriminator(decoded_images)

        ## Gener ##
        reconstruction_loss = self.perc_loss(target, decoded_images)
        gan_loss = -torch.mean(disc_fake)
        lossG = reconstruction_loss + vq_loss + self.val_gan_factor*gan_loss

        ## Discr ##
        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        lossD = 0.5 * (d_loss_real + d_loss_fake)

        mets = self.val_metrics(target, decoded_images.clamp(-1,1))

        self.log("val_loss", lossG, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_lossD", lossD, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        for k,v in mets.items():
            self.log(k, v, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def on_train_epoch_end(self):

        batch = next(iter(self.test_dataloader))
        I0, target, I1 = batch.transpose(0,1)
        I0, target, I1 = I0.to(self.device), target.to(self.device), I1.to(self.device)

        decoded_images, _ = self.forward(target, I0, I1, in_trainig = False)

        norm_I0 = denorm(I0, self.mean, self.sd)
        norm_I1 = denorm(I1, self.mean, self.sd)
        norm_target = denorm(target, self.mean, self.sd)
        norm_decoded = denorm(decoded_images.clamp(-1, 1), self.mean, self.sd)

        #save_images((norm_decoded, norm_target), f"./_outputs/vqgan/target_{self.current_epoch}.png", nrow=3)
        save_triplet([norm_I0, norm_decoded, norm_target, norm_I1], f"./_outputs/encoder_decoder/target_{self.current_epoch}.png", nrow=1)
        if self.use_ema:
            torch.save(self.ema_vqfigan, f"./_checkpoints/vqvae/model_{self.current_epoch}.ckpt")
        else:
            torch.save(self.vqfigan, f"./_checkpoints/vqvae/model_{self.current_epoch}.ckpt")

    def configure_optimizers(self):
        lr_g = 8e-05
        lr_d = 1e-04
        g_optimizer = optim.Adam(self.vqfigan.parameters(), lr=lr_g, betas=(0.5, 0.9))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
        optimizers = [g_optimizer, d_optimizer]

        scheduler = [
            {
                'scheduler': CosineAnnealingLR(
                    g_optimizer,
                    T_max=1000,
                    eta_min=1e-5
                ),
                'interval': 'step',
                'frequency': 1
            },
            {
                'scheduler': CosineAnnealingLR(
                    d_optimizer,
                    T_max=800,
                    eta_min=1e-5
                ),
                'interval': 'step',
                'frequency': 1
            },
        ]
        
        return optimizers, scheduler

