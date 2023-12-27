from imports.common_imports import *

from utils.basic_layers import ResnetBlock, DownSampleBlock, UpSampleBlock, GroupNorm 
from utils.max_vit_att import MaxViTAtt, CrossViTAtt

from cupy_module import dsepconv

def exist(val):
    return val is not None

class TempPosEncoding(nn.Module):
    def __init__(self, time_dim, flow_estimation = 'farneback'):
        super(TempPosEncoding, self).__init__()

        assert flow_estimation in ['farneback', 'tvl1'], 'flow_estimation must be farneback or tvl1'
        if flow_estimation == 'farneback': 
            self.flow_estimation = self.OptFlow_Farneback
        elif flow_estimation == 'tvl1':
            self.flow_estimation = self.OptFlow_DualTVL1

        self.time_dim = time_dim

    def OptFlow_Farneback(self, I0, I1):
        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(I0_gray, I1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)

    def OptFlow_DualTVL1(self, I0, I1,
        tau: float = 0.25,
        lambda_: float = 0.15,
        theta: float = 0.3,
        scales_number: int = 5,
        warps: int = 5,
        epsilon: float = 0.01,
        inner_iterations: int = 30,
        outer_iterations: int = 10,
        scale_step: float = 0.8,
        gamma: float = 0.0
    ):
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        optical_flow.setTau(tau)
        optical_flow.setLambda(lambda_)
        optical_flow.setTheta(theta)
        optical_flow.setScalesNumber(scales_number)
        optical_flow.setWarpingsNumber(warps)
        optical_flow.setEpsilon(epsilon)
        optical_flow.setInnerIterations(inner_iterations)
        optical_flow.setOuterIterations(outer_iterations)
        optical_flow.setScaleStep(scale_step)
        optical_flow.setGamma(gamma)

        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = optical_flow.calc(I0_gray, I1_gray, None)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)

    def morph_open(self, x, k):
        if k==0:
            return x
        else:
            with torch.no_grad():
                return kornia.morphology.opening(x, torch.ones(k,k,device=x.device))

    def TIMDEX(self, I0, It, I1, k = 5, threshold = 2e-2):
        flow0tot = self.flow_estimation(I0, It)
        flow1tot = self.flow_estimation(It, I1)
    
        transform = transforms.Grayscale()
        I0_gray = transform(I0)
        It_gray = transform(It)
        I1_gray = transform(I1)

        mask0tot = self.morph_open(It_gray - I0_gray, k=k)
        mask1tot = self.morph_open(I1_gray - It_gray, k=k)

        mask0tot = (abs(mask0tot) > threshold).to(torch.uint8)
        mask1tot = (abs(mask1tot) > threshold).to(torch.uint8)
        
        d0tot = torch.sum(torch.norm(flow0tot*mask0tot, dim=1), dim = (1, 2)) 
        d1tot = torch.sum(torch.norm(flow1tot*mask1tot, dim=1), dim = (1, 2))
        
        return d0tot / (d0tot + d1tot + 1e-8)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, I0, It, I1):
        t = self.TIMDEX(I0, It, I1)
        t = t.unsqueeze(1)
        t = self.pos_encoding(t, self.time_dim)
        return t

class Encoder(nn.Module):
    def __init__(self, latent_chanels):
        super(Encoder, self).__init__()
        in_c = 3
        c = 128
        channels = [c, c, 2*c, 2*c, 2*c, 4*c]
        att_channels = [False, False, False, False, False]
        assert len(channels) - 1 == len(att_channels), "Number of attention blocks must be equal to number of step blocks (steps = channels - 1)"  
        num_head_channels = 64
        self.n_iter = len(channels) - 1

        self.init_conv = nn.Conv2d(in_c, channels[0], kernel_size=3, stride=1,padding=1)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResnetBlock(channels[i], channels[i + 1], kernel_size=3, temb_channels=0),
                DownSampleBlock(channels[i + 1], with_conv=True, deformable=False),
            ) for i in range(self.n_iter)
        ])

        self.mid_att_block = nn.Sequential(
            ResnetBlock(channels[-1], channels[-1], temb_channels=0),
            MaxViTAtt(heads = channels[-1] // num_head_channels, dim_head = num_head_channels),
            ResnetBlock(channels[-1], channels[-1], temb_channels=0),
        )

        self.conv_to_latent = nn.Conv2d(channels[-1], latent_chanels, kernel_size=3, stride=1,padding=1)

    def forward(self, x, I0, I1):
        extremes_cond = exist(I0) and exist(I1)
        target_cond = exist(x)

        if target_cond:
            x = self.init_conv(x)
        if extremes_cond:
            phi0 = self.init_conv(I0)
            phi1 = self.init_conv(I1)
            features = []                                     
        for i in range(self.n_iter):
            if target_cond:
                x = self.res_blocks[i](x)
            if extremes_cond:
                phi0 = self.res_blocks[i](phi0)
                phi1 = self.res_blocks[i](phi1)
                features.append(torch.cat((phi0, phi1), dim = 1))

        if target_cond:
            x = self.mid_att_block(x) 
            x = self.conv_to_latent(x)
        if extremes_cond:
            phi0 = self.mid_att_block(phi0)
            phi1 = self.mid_att_block(phi1)
            phi0 = self.conv_to_latent(phi0)
            phi1 = self.conv_to_latent(phi1)

        if target_cond and extremes_cond:
            return x, phi0, phi1, features
        if target_cond and not extremes_cond:
            return x
        if not target_cond and extremes_cond:
            return phi0, phi1, features

class VQLayer(nn.Module):
    def __init__(
            self,
            num_codebook_vectors = 2040, 
            emb_dim = 4,
            beta = 0.25
    ):
        super(VQLayer, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        vq_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, vq_loss

class Decoder(nn.Module):
    def __init__(self, latent_chanels):
        super(Decoder, self).__init__()
        c = 128
        c_out = 64
        channels = [4*c, 4*c, 2*c, 2*c, 2*c, c]
        att_channels = [False, False, False, False, False]
        assert len(channels) - 1 == len(att_channels), "Number of attention blocks must be equal to number of step blocks (steps = channels - 1)"  
        num_head_channels = 64
        self.n_iter = len(channels) - 1

        self.conv_from_latent = nn.Conv2d(latent_chanels, channels[0], kernel_size=3, stride=1,padding=1)

        self.mid_att_block = nn.Sequential(
            ResnetBlock(channels[0], channels[0], temb_channels=0),
            MaxViTAtt(heads = channels[0] // num_head_channels, dim_head = num_head_channels),
            ResnetBlock(channels[0], channels[0], temb_channels=0),
        )

        self.res_blocks1 = nn.ModuleList([ 
            ResnetBlock(channels[i], channels[i + 1], kernel_size=3, temb_channels=0)\
                for i in range(self.n_iter)
        ])
        self.cross_att_block = nn.ModuleList([
            CrossViTAtt(channels[i + 1], 2*channels[i + 1], heads = channels[i + 1] // num_head_channels, dim_head = num_head_channels)\
                for i in range(self.n_iter)
        ])
        self.up_blocks = nn.ModuleList([
            UpSampleBlock(channels[i + 1], with_conv=True) for i in range(self.n_iter)                  
        ])

        self.norm_out = GroupNorm(channels[-1])
        self.nonlinearity = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], c_out, kernel_size=3, stride=1,padding=1)

    def forward(self, x, features):
        x = self.conv_from_latent(x)
        x = self.mid_att_block(x)
        for i, i_rev in enumerate(reversed(range(self.n_iter))):
            x = self.res_blocks1[i](x)
            x = self.cross_att_block[i](x, features[i_rev])  
            x = self.up_blocks[i](x)
        x = self.nonlinearity(self.norm_out(x))
        x = self.conv_out(x)
        return x

class DeformConvInterpKernels(nn.Module):
    def __init__(self, c_in=64):
        super(DeformConvInterpKernels, self).__init__()

        self.moduleAlpha1 = self.OffsetHead(c_in)
        self.moduleAlpha2 = self.OffsetHead(c_in)
        self.moduleBeta1 = self.OffsetHead(c_in)
        self.moduleBeta2 = self.OffsetHead(c_in)
        self.moduleKernelHorizontal1 = self.KernelHead(c_in)
        self.moduleKernelHorizontal2 = self.KernelHead(c_in)
        self.moduleKernelVertical1 = self.KernelHead(c_in)
        self.moduleKernelVertical2 = self.KernelHead(c_in)
        self.moduleMask1 = self.MaskHead(c_in)
        self.moduleMask2 = self.MaskHead(c_in)
        self.moduleResidual = self.ResidualHead(c_in)
        self.modulePad = nn.ReplicationPad2d([2, 2, 2, 2])
    
    def KernelHead(self, c_in):
        return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
            )

    def OffsetHead(self, c_in):
        return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=32, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=5 ** 2, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1)
            )

    def MaskHead(self, c_in):
        return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=32, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=5 ** 2, out_channels=5 ** 2, kernel_size=3,
                            stride=1, padding=1),
                nn.Sigmoid()
            )

    def ResidualHead(self, c_in):
        return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            )
    
    def forward(self, x, I0, I1):
        alpha1 = self.moduleAlpha1(x)
        alpha2 = self.moduleAlpha2(x)
        beta1 = self.moduleBeta1(x)
        beta2 = self.moduleBeta2(x)
        v1 = self.moduleKernelVertical1(x)
        v2 = self.moduleKernelVertical2(x)
        h1 = self.moduleKernelHorizontal1(x)
        h2 = self.moduleKernelHorizontal2(x)
        mask1 = self.moduleMask1(x)
        mask2 = self.moduleMask2(x)
        warped1 = dsepconv.FunctionDSepconv(self.modulePad(I0), v1, h1, alpha1, beta1, mask1)
        warped2 = dsepconv.FunctionDSepconv(self.modulePad(I1), v2, h2, alpha2, beta2, mask2)
        warped = warped1 + warped2
        out = warped + self.moduleResidual(x)
        return out

class VQFIGAN(nn.Module):
    def __init__(self):  
        super(VQFIGAN, self).__init__()

        latent_chanels = 3
    
        self.pos_encoding = TempPosEncoding(128, flow_estimation = 'farneback')

        self.encoder = Encoder(latent_chanels)
        self.vqlayer = VQLayer(
                num_codebook_vectors = 8129, 
                emb_dim = latent_chanels,
                beta = 0.25
            )
        self.decoder = Decoder(latent_chanels)
        self.deform_conv_interp_kernels = DeformConvInterpKernels()

    def forward(self, x, I0, I1):
        x, _, _, features = self.encoder(x, I0, I1)
        x, vq_loss = self.vqlayer(x)
        x = self.decoder(x, features)
        x = self.deform_conv_interp_kernels(x, I0, I1)
        return x, vq_loss

    def encode_target(self, x):
        x = self.encoder(x, None, None)
        return x

    def encode_features(self, I0, I1):
        phi0, phi1, features = self.encoder(None, I0, I1)
        return phi0, phi1, features
    
    def decode(self, z, I0, I1, features = None):
        if not exist(features):
            _, _, features = self.encode_features(I0, I1)
        z, _ = self.vqlayer(z)
        It = self.decoder(z, features)
        It = self.deform_conv_interp_kernels(It, I0, I1)
        return It


