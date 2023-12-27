from imports.common_imports import *
from utils.basic_layers import DeformableConv2d

# Code adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py

def exist(val):
    return val is not None

# Basic layers
class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, c=None):
        if exist(c):
            return self.fn(self.norm(x), self.norm(c)) + x
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_rate = 4, dropout = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_rate)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvBock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate = 4,
            shrinkage_rate = 0.25
    ):
        super(MBConvBock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = int(in_channels * expansion_rate)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding = 1, groups = hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            #SqueezeExcitation(hidden_channels, shrinkage_rate = shrinkage_rate),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.skip(x)
        else:
            return self.block(x) + x

# Attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x, c = None):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        if exist(c):
            c = self.norm(c)
            c = rearrange(c, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
            q = self.to_q(x)
            k = self.to_k(c)
            v = self.to_v(c)
        else:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # split heads

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) # sim
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')# add positional bias
        attn = self.attend(sim) # attention
        out = einsum('b h i j, b h j d -> b h i d', attn, v) # aggregate

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width) # merge heads
        out = self.to_out(out) # combine heads out
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class MaxViTAtt(nn.Module):
    def __init__(
        self,
        heads = 8,
        dim_head = 64,
        window_size = 8,
        dropout = 0.0,
    ):
        super(MaxViTAtt, self).__init__()
        w = window_size
        layer_dim = dim_head * heads
        self.vitblock = nn.Sequential(
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
            Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
            Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
            Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
            Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
        )

    def forward(self, x):
        x = self.vitblock(x)
        return x


class CrossViTAtt(nn.Module):
    def __init__(
        self,
        channels,   
        context_channels,
        heads = 8,
        dim_head = 64,
        window_size = 4,
        deformable = False,
        dropout = 0.0,
    ):
        super(CrossViTAtt, self).__init__()
        w = window_size
        layer_dim = dim_head * heads
        
        self.context_channels = context_channels
        self.channels = channels
        if channels != context_channels:
            if deformable:
                self.context_skip = DeformableConv2d(context_channels, channels, kernel_size=5, stride=1, padding=2)
            else:
                self.context_skip = nn.Conv2d(context_channels, channels, kernel_size=1, stride=1, padding=0)   

        self.rearrange_window_forward = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)  # block-like attention
        self.res_att_window = Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.res_ff_window = Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_window_backward = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.rearrange_grid_forward = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)  # grid-like attention
        self.res_att_grid = Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.res_ff_grid = Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_grid_backward = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')

    def forward(self, x, c):

        if self.channels != self.context_channels:
            c = self.context_skip(c)

        x = self.rearrange_window_forward(x)
        c = self.rearrange_window_forward(c)
        x = self.res_att_window(x, c)
        x = self.res_ff_window(x)
        x = self.rearrange_window_backward(x)
        c = self.rearrange_window_backward(c)

        x = self.rearrange_grid_forward(x)
        c = self.rearrange_grid_forward(c)
        x = self.res_att_grid(x, c)
        x = self.res_ff_grid(x)
        x = self.rearrange_grid_backward(x)

        return x
