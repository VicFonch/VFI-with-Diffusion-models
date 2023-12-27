from imports.common_imports import *
import sys
from diffusion.att_unet import UNet
from diffusion.diffusion import Diffusion
#from diffusion.TIMDEX import TIMDEXPosEncoding

from utils.utils import *
from utils.ema import EMA
 
class TrainerDiffusion(pl.LightningModule):
    def __init__(self, test_dataloader, sampling_process="ddpm", use_ema=False):
        super(TrainerDiffusion, self).__init__()

        assert sampling_process in ["ddim", "ddpm"], "sampling_process must be 'ddim' or 'ddpm'"
        self.sampling_process = sampling_process

        self.mean, self.sd = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        self.test_dataloader = test_dataloader
        
        checkpoint_path = "./_models/vqfigan/vqfigan.ckpt"
        self.vqfigan = torch.load(checkpoint_path).eval().requires_grad_(False)

        #self.timdex = TIMDEXPosEncoding(128, flow_estimation = 'farneback').requires_grad_(False)\

        noise_steps = 1000 if sampling_process == "ddpm" else 200
        self.diffusion = Diffusion(noise_steps)
        
        self.unet = UNet()

        self.MSE = nn.MSELoss()
        
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(beta=0.995)
            self.ema_unet =  copy.deepcopy(self.unet).eval().requires_grad_(False)

        self.scheduler_val_loss = [] 

    def forward(self, target, I0, I1):
        z, z0, z1, _ = self.vqfigan.encoder(target, I0, I1)

        #timdex = self.timdex(I0, target, I1)
        t = self.diffusion.sample_timesteps(z.shape[0]).to(self.device)
        z_t, noise = self.diffusion.noise_images(z, t)
        predicted_noise = self.unet(z_t, z0, z1, t)

        return predicted_noise, noise

    def training_step(self, batch, batch_idx):
        I0, target, I1 = batch.transpose(1, 0)
        predicted_noise, noise = self.forward(target, I0, I1)

        loss = self.MSE(predicted_noise, noise)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
        if self.use_ema:
            self.ema.step_ema(self.ema_unet, self.unet)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        I0, target, I1 = batch.transpose(1, 0)
        predicted_noise, noise = self.forward(target, I0, I1)
        
        loss = self.MSE(predicted_noise, noise)
        self.scheduler_val_loss.append(loss) 
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
    @torch.no_grad()
    def on_train_epoch_end(self):

        batch = next(iter(self.test_dataloader))
        I0, target, I1 = batch.transpose(0,1)
        I0, target, I1 = I0.to(self.device), target.to(self.device), I1.to(self.device)
        z0, z1, features = self.vqfigan.encode_features(I0, I1)

        if self.sampling_process == "ddim":
            if self.use_ema: sampled_images = self.diffusion.ddim_sample(self.ema_unet, z0, z1)
            else: sampled_images = self.diffusion.ddim_sample(self.unet, z0, z1)
        else: 
            if self.use_ema: sampled_images = self.diffusion.ddpm_sample(self.ema_unet, z0, z1)
            else: sampled_images = self.diffusion.ddpm_sample(self.unet, z0, z1)

        sampled_images = self.vqfigan.decode(sampled_images, I0, I1, features)

        norm_I0 = denorm(I0, self.mean, self.sd)
        norm_I1 = denorm(I1, self.mean, self.sd)
        norm_target = denorm(target, self.mean, self.sd)
        norm_sampled_images = denorm(sampled_images.clamp(-1, 1), self.mean, self.sd)

        save_triplet([norm_I0, norm_sampled_images, norm_target, norm_I1], f"_outputs/diffusion/target_{self.current_epoch}.png", nrow=1)
        if self.use_ema:
            torch.save(self.ema_unet, os.path.join("_checkpoints/unet",  f"model_{self.current_epoch}.ckpt"))
        else:
            torch.save(self.unet, os.path.join("_checkpoints/unet",  f"model_{self.current_epoch}.ckpt"))

    def configure_optimizers(self):
        lr = 8e-06
        optimizer = optim.AdamW(self.unet.parameters(), lr=lr)
        scheduler = [
                {
                    'scheduler': CosineAnnealingLR(
                        optimizer,
                        T_max=1000,
                        eta_min=1e-6
                    ),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
        
        return [optimizer,], scheduler

