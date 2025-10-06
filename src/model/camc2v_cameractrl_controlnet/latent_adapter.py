import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from einops import rearrange
from inspect import isfunction

from termcolor import colored


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x
        
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)




class EpipolarCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64,
                 dropout=0.0, num_register_tokens=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.context_dim = context_dim
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.dropout = dropout
        self.out_dim = out_dim
        if out_dim is not None:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.forward = self.efficient_forward
        self.num_register_tokens = num_register_tokens

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn((1, num_register_tokens, context_dim)), requires_grad=True)


    def forward(self, x, attn_mask=None):
        return x

    def efficient_forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor = None):
        '''
        :param x:       B,L1,C
        :param context:       B,L2,C
        :param attn_mask: B,L1,L2
        :return:
        '''
        #import ipdb; ipdb.set_trace()
        q = self.to_q(x)
        B = q.shape[0]

        if self.num_register_tokens > 0:
            context = torch.concat([self.register_tokens.repeat(B, 1, 1), context], dim=1)  # B, L2, D --> B, num_registers+L2, D

        k = self.to_k(context)
        v = self.to_v(context)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.nn.functional.pad(attn_mask, (self.num_register_tokens,0), mode='constant', value=True) # B,L1,L2 --> B,L1, num_registers+L2
            else:
                attn_mask = torch.nn.functional.pad(attn_mask, (self.num_register_tokens,0), mode='constant', value=-0.0) # B,L1,L2 --> B,L1, num_registers+L2

        q, k, v = map(lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None)
        out = rearrange(out, "B H L D -> B L (H D)")

        return self.to_out(out)
    

class LatentAdapterBlock(nn.Module):

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 context_dim: int = None,
                 num_heads: int = 8):
        super().__init__()

        context_dim = context_dim or input_dim

        self.proj_in_x = nn.Linear(input_dim, hidden_dim) 
        self.proj_in_context = nn.Linear(context_dim, hidden_dim) 
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.resnet_block = ResnetBlock(in_c=hidden_dim, out_c=hidden_dim, down=False, ksize=3, sk=True, use_conv=True)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)# kdim=input_dim, vdim=input_dim)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)# kdim=input_dim, vdim=input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm2_context = nn.LayerNorm(hidden_dim)
        self.cross_attention = EpipolarCrossAttention(query_dim=hidden_dim, context_dim=hidden_dim, heads=num_heads)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.proj_out = nn.Linear(hidden_dim, output_dim) if output_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward function of latent adapter block.

        Args:
        :param x: Input tensor of shape (batch_size, channels, seq_len, height, width).
        :param context: Context tensor of shape (batch_size, channels, num_condition, height, width).
        :param mask: Optional attention mask of shape (batch_size, seq_len, context_len).
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        context = rearrange(context, 'b c n h w -> b (n h w) c')
        #import ipdb; ipdb.set_trace()

        # Projection in
        x = self.proj_in_x(x)
        context = self.proj_in_context(context)
        # ResNet Block
        x = self.norm1(x)
        x = rearrange(x, 'b (f h w) c -> (b f) c h w', f=T, h=H, w=W)
        x = self.resnet_block(x)
        # Spatial Attention
        x = rearrange(x, '(b f) c h w -> (b f) (h w) c', f=T, h=H, w=W)
        x = x + self.spatial_attention(x, x, x)[0]
        # Temporal Attention
        x = rearrange(x, '(b f) (h w) c -> (b h w) (f) c', f=T, h=H, w=W)
        x = x + self.temporal_attention(x, x, x)[0]
        # Cross Attention
        x = rearrange(x, '(b h w) (f) c -> b (f h w) c', f=T, h=H, w=W)
        x = self.norm2(x)
        context = self.norm2_context(context)
        x = x + self.cross_attention(x, context, attn_mask=mask)[0]
        # Feed Forward
        x = self.norm3(x)
        x = x + self.feed_forward(x)
        x = self.proj_out(x)
        x = rearrange(x, 'b (f h w) c -> b c f h w', f=T, h=H, w=W)
        return x


class LatentAdapter(nn.Module):

    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 4,
                 num_blocks: int = 4,
                 hidden_dim: int = 512,
                 checkpoint: str = None
                 ):
        super().__init__()
        self.blocks = []

        for i in range(num_blocks):
            idim = input_dim if i == 0 else hidden_dim
            odim = output_dim if i == num_blocks - 1 else hidden_dim
            block = LatentAdapterBlock(input_dim=idim, output_dim=odim, hidden_dim=hidden_dim, context_dim=input_dim)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
        self.zero_conv = nn.Conv3d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.zero_conv.weight.data.zero_()
        self.zero_conv.bias.data.zero_()
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)  

    def load_checkpoint(self, checkpoint: str):
        state_dict = torch.load(checkpoint, map_location='cpu')["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        print(colored(f"=> loaded latent adapter checkpoint from: {checkpoint}", "green"))

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward function of latent adapter.

        Args:
        :param x: Input tensor of shape (batch_size, channel, num_frames, height, width).
        :param context: Context tensor of shape (batch_size, channel, num_conditions, height, width).
        :param mask: Optional attention mask of shape (batch_size, seq_len, context_len).

        """
        #import ipdb; ipdb.set_trace()
        #T, H, W = x.shape[2:]
        #x = rearrange(x, 'b c f h w -> b (f h w) c')
        #context = rearrange(context, 'b c n h w -> b (n h w) c')
        inp = x.clone()
        for block in self.blocks:
            x = block(x, context, mask)

        #x = rearrange(x, 'b c f h w -> b f c h w')
        x = inp + self.zero_conv(x)
        #x = rearrange(x, 'b (f h w) c -> b c f h w', f=T, h=H, w=W)
        return x


if __name__ == "__main__":
    from VidUtil.torch_utils import model_summary
    #import ipdb; ipdb.set_trace()
    model = LatentAdapter(input_dim=4, output_dim=4, num_blocks=4, hidden_dim=512).to('cuda')
    print(model_summary(model))
    #import ipdb; ipdb.set_trace()
    x = torch.randn((2, 4, 16, 32, 32)).to('cuda')
    context = torch.randn((2, 4, 3, 32, 32)).to('cuda')

    print(f"x: {x.shape}\ncontext: {context.shape}")
    with torch.no_grad():
        for i in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            out = model(x, context, None)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Time: {end_time - start_time} seconds")

    print(out.shape)
