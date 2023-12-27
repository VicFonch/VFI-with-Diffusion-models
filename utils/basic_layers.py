from imports.common_imports import *

def exist(val):
    return val is not None

class GroupNorm(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    def __init__(self, beta = 0.2):
        super(Swish, self).__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.beta*x)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(
            in_channels, 
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size, 
            stride=stride,
            padding=self.padding, 
            dilation=self.dilation, 
            bias=True    
        )

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias
        )
        
        self.weight = self.regular_conv.weight

    def forward(self, x):
        offset = self.offset_conv(x) 
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
            dilation=self.dilation
        )
        
        return x

class ResnetBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size = 3,
            norm = "group", 
            deformable = False,
            dropout = 0.0,
            conv_shortcut=False, 
            temb_channels=256
    ):
        assert norm in ["group", "instance", "batch", "layer"]
        
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm == "group": normalize = GroupNorm
        elif norm == "batch": normalize = nn.BatchNorm2d
        elif norm == "instance": normalize = nn.InstanceNorm2d
        elif norm == "layer": normalize = nn.LayerNorm

        if deformable: conv2d = DeformableConv2d
        else: conv2d = nn.Conv2d 
        
        padding = int((kernel_size - 1) / 2)

        self.block1 = nn.Sequential(
                normalize(in_channels),
                nn.SiLU(),
                conv2d(in_channels,out_channels,kernel_size=kernel_size, stride=1, padding=padding),
            )
            
        if temb_channels > 0:
            self.nonlinearity = nn.SiLU()
            self.temb_proj = nn.Linear(temb_channels,out_channels)

        self.block2 = nn.Sequential(
                normalize(out_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding),
            )
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x, temb = None):
        h = x
        h = self.block1(h)
        if temb is not None:
            h = h + self.temb_proj(self.nonlinearity(temb))[:,:,None,None]
        h = self.block2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv = True, deformable = False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            if deformable:
                self.conv = DeformableConv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv = True, deformable = False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            if deformable:
                self.conv = DeformableConv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=0)
            else:
                self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    

class NonLocalAttention(nn.Module):
    def __init__(self, channels):
        super(NonLocalAttention, self).__init__()
        self.in_channels = channels
    
        self.gn = GroupNorm(channels)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A

class NonLocalCrossAttention(nn.Module):
    def __init__(self, channels, context_channels):
        super(NonLocalCrossAttention, self).__init__()
        self.in_channels = channels
        
        self.gn = GroupNorm(channels)
        if channels != context_channels:
            self.cond_norm = nn.Conv2d(context_channels, channels, 1, 1, 0),
            
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, cond = None):
        h_ = self.gn(x)
        cond = self.gn(cond) if exist(cond) else None
        if exist(cond):
            if x.shape[1] != cond.shape[1]:
                cond = self.cond_norm(cond)
            cond = self.gn(cond)
            q = self.to_q(h_)
            k = self.to_k(cond)
            v = self.to_v(cond)
        else:
            q = self.to_q(h_)
            k = self.to_k(h_)
            v = self.to_v(h_)

        batch, channels, size, size = q.shape

        q = q.reshape(batch, channels, size**2)
        q = q.permute(0, 2, 1)
        k = k.reshape(batch, channels, size**2)
        v = v.reshape(batch, channels, size**2)

        attn = torch.bmm(q, k)
        attn = attn * (int(channels)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(batch, channels, size, size)

        return x + A

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class CrossAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size

        self.ln = nn.LayerNorm([channels])
        self.cond_norm = nn.Conv2d(2*channels, channels, 1, 1, 0),
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, c):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        if exist(c):
            c = self.cond_norm(c)
            c_ln = c.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
            attention_value, _ = self.mha(x_ln, c_ln, c_ln)
        else:
            attention_value, _ = self.mha(x_ln, x_ln, x_ln)

        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


