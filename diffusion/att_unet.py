from imports.common_imports import *

from utils.basic_layers import ResnetBlock, DownSampleBlock, UpSampleBlock
from utils.max_vit_att import MaxViTAtt

class UNet(nn.Module):
    def __init__(self, c_in=9, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        c = 320

        self.inc = nn.Conv2d(c_in, c, kernel_size=3, padding=1)

        self.res1 = ResnetBlock(c, c, temb_channels=time_dim)
        self.max_vit1 = MaxViTAtt(heads = 4, dim_head = c//4, window_size = 2)
        self.res2 = ResnetBlock(c, c, temb_channels=time_dim)
        self.max_vit2 = MaxViTAtt(heads = 4, dim_head = c//4, window_size = 2)

        self.down1 = DownSampleBlock(c)

        self.res3 = ResnetBlock(c, 2*c, temb_channels=time_dim)
        self.max_vit3 = MaxViTAtt(heads = 4, dim_head = 2*c//4, window_size = 2)
        self.res4 = ResnetBlock(2*c, 2*c, temb_channels=time_dim)
        self.max_vit4 = MaxViTAtt(heads = 4, dim_head = 2*c//4, window_size = 2)

        self.down2 = DownSampleBlock(2*c)

        self.res5 = ResnetBlock(2*c, 4*c, temb_channels=time_dim)
        self.max_vit5 = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)
        self.res6 = ResnetBlock(4*c, 4*c, temb_channels=time_dim)
        self.max_vit6 = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)

        self.bot1_res = ResnetBlock(4*c, 4*c, temb_channels=time_dim)
        self.bot1_att = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)
        self.bot2_res = ResnetBlock(4*c, 4*c, temb_channels=time_dim)
        self.bot2_att = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)

        self.res7 = ResnetBlock(4*c, 4*c, temb_channels=time_dim)
        self.max_vit7 = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)
        #concat 8c to 4c
        self.res8 = ResnetBlock(8*c, 4*c, temb_channels=time_dim)
        self.max_vit8 = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)

        #concat 6c to 4c
        self.res9 = ResnetBlock(6*c, 4*c, temb_channels=time_dim)
        self.max_vit9 = MaxViTAtt(heads = 4, dim_head = 4*c//4, window_size = 2)
        self.up1 = UpSampleBlock(4*c)

        #concat 6c to 2c
        self.res10 = ResnetBlock(6*c, 2*c, temb_channels=time_dim)
        self.max_vit10 = MaxViTAtt(heads = 4, dim_head = 2*c//4, window_size = 2)
        #concat 4c to 2c
        self.res11 = ResnetBlock(4*c, 2*c, temb_channels=time_dim)
        self.max_vit11 = MaxViTAtt(heads = 4, dim_head = 2*c//4, window_size = 2)

        #concat 3c to 2c
        self.res12 = ResnetBlock(3*c, 2*c, temb_channels=time_dim)
        self.max_vit12 = MaxViTAtt(heads = 4, dim_head = 2*c//4, window_size = 2)
        self.up2 = UpSampleBlock(2*c)

        #concat 3c to c
        self.res13 = ResnetBlock(3*c, c, temb_channels=time_dim)
        self.max_vit13 = MaxViTAtt(heads = 4, dim_head = c//4, window_size = 2)
        #concat 2c to c
        self.res14 = ResnetBlock(2*c, c, temb_channels=time_dim)
        self.max_vit14 = MaxViTAtt(heads = 4, dim_head = c//4, window_size = 2)

        self.outc = nn.Conv2d(c, c_out, kernel_size=3, padding=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, z, z0, z1, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x = torch.cat([z, z0, z1], dim=1) 

        x1 = self.inc(x)
        
        x2 = self.res1(x1, t)
        x2 = self.max_vit1(x2)

        x3 = self.res2(x2, t)
        x3 = self.max_vit2(x3)

        x4 = self.down1(x3)
        
        x5 = self.res3(x4, t)
        x5 = self.max_vit3(x5)

        x6 = self.res4(x5, t)
        x6 = self.max_vit4(x6)

        x7 = self.down2(x6)

        x8 = self.res5(x7, t)
        x8 = self.max_vit5(x8)

        x = self.res6(x8, t)
        x = self.max_vit6(x)

        x = self.bot1_res(x, t)
        x = self.bot1_att(x)
        x = self.bot2_res(x, t)
        x = self.bot2_att(x)

        x = self.res7(x, t)
        x = self.max_vit7(x)
        #concat 8c to 4c
        x = self.res8(torch.cat([x, x8], dim=1), t)
        x = self.max_vit8(x)
        
        #concat 6c to 4c
        x = self.res9(torch.cat([x, x7], dim=1), t)
        x = self.max_vit9(x)
        x = self.up1(x)

        #concat 6c to 2c
        x = self.res10(torch.cat([x, x6], dim=1), t)
        x = self.max_vit10(x)
        #concat 4c to 2c
        x = self.res11(torch.cat([x, x5], dim=1), t)
        x = self.max_vit11(x)

        #concat 3c to 2c
        x = self.res12(torch.cat([x, x4], dim=1), t)
        x = self.max_vit12(x)
        x = self.up2(x)

        #concat 3c to c
        x = self.res13(torch.cat([x, x3], dim=1), t)
        x = self.max_vit13(x)
        #concat 2c to c
        x = self.res14(torch.cat([x, x2], dim=1), t)
        x = self.max_vit14(x)

        output = self.outc(x)
        
        return output