from functools import partial
from abc import abstractmethod
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import os, sys
print(os.getcwd())
sys.path.append(os.getcwd())
# project utilities
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.common import checkpoint
from lvdm.basics import (
    zero_module,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization
)
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class ZeroConvolution(nn.Module):
    """
    A convolution with zero initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to children that support it.
    Also routes context to (Spatial|Temporal)Transformer layers, and reshapes for
    temporal transformers when needed.
    """
    def forward(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                # (bt, c, h, w) -> (b, c, t, h, w) for temporal attention
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer with optional convolution.
    If dims == 3, downsampling occurs in the inner two spatial dims.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block that optionally changes channels and (optionally) applies temporal conv.
    It can also do up/down when used inside an encoder/decoder, but here we only use `down`
    in the encoder when resblock_updown=True.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            # not used in the encoder path
            self.h_upd = Upsample(channels, False, dims)  # noqa: F821 (defined in decoder files normally)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware
            )

    def forward(self, x, emb, batch_size=None):
        """
        :param x: [(b*t), C, H, W]
        :param emb: [(b*t), emb_channels] or [b, emb_channels] depending on caller
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple, self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb, batch_size=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TemporalConvBlock(nn.Module):
    """
    Lightweight temporal convolution block (3D convs) used inside ResBlock when enabled.
    Adapted from ModelScope video UNet.
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))

        # initialize last conv to zero so the block starts as identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return identity + x


class ContextEncoder(nn.Module):
    """
    Encoder + (optional) middle block extracted from the original UNetModel.

    Builds:
      - time embedding (and optional fps embedding)
      - input_blocks (downsampling tower with optional spatial/temporal attention)
      - an optional `init_attn` before the first block when `addition_attention=True`
      - middle_block (ResBlock -> [Spatial, Temporal] -> ResBlock)

    Excludes:
      - decoder/output blocks and final conv head

    Forward returns:
      - h_enc: encoded feature after the middle block, shape [(b*t), C, H, W]
      - hs: list of skip features collected from input blocks (each is [(b*t), C, H, W])
      - input_ds: list of downsample factors per collected feature (mirrors original)
    """
    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,  # kept for parity with original signature (unused here)
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 context_dim=None,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 transformer_depth=1,
                 use_linear=False,
                 use_checkpoint=False,
                 temporal_conv=False,
                 tempspatial_aware=False,
                 temporal_attention=True,
                 use_relative_position=True,
                 use_causal_attention=False,
                 temporal_length=None,
                 use_fp16=False,
                 addition_attention=False,
                 temporal_selfatt_only=True,
                 image_cross_attention=False,
                 image_cross_attention_scale_learnable=False,
                 include_middle_block=False,
                 default_fs=4,
                 checkpoint: str = None,
                 fs_condition=False,
                 zero_convolution: bool = True,
                 activation_ckpt: bool = True,
                 layer_fusion_strategy: str = 'max',  # kept for parity (unused here)
                 ):
        super(ContextEncoder, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.layer_fusion_strategy = layer_fusion_strategy  # not used in encoder
        self.activation_ckpt = activation_ckpt

        # Time embedding (and optional FPS conditioning)
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # Input stem
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )
        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels if num_head_channels != -1 else model_channels // num_heads,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=False,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                    ds=1
                )
            )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.input_ds, self.output_ds = [ds], []  # output_ds kept for parity; not used here

        # Down path with attention at configured resolutions
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims,
                        use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware, use_temporal_conv=temporal_conv
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head,
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False,
                            video_length=temporal_length, image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                            is_output_block=False, ds=ds
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only,
                                causal_attention=use_causal_attention, relative_position=use_relative_position,
                                temporal_length=temporal_length, is_output_block=False, ds=ds
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_ds.append(ds)
                input_block_chans.append(ch)

            # Downsample between levels (either residual up/down or explicit downsample)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, down=True
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                self.input_ds.append(ds)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle block (kept exactly as in original)
        self.middle_block = None
        if include_middle_block:
            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels

            middle_layers = [
                ResBlock(
                    ch, time_embed_dim, dropout,
                    dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                    use_temporal_conv=temporal_conv
                ),
                SpatialTransformer(
                    ch, num_heads, dim_head,
                    depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                    use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length,
                    image_cross_attention=self.image_cross_attention,
                    image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                    is_output_block=False, ds=ds
                )
            ]
            if self.temporal_attention:
                middle_layers.append(
                    TemporalTransformer(
                        ch, num_heads, dim_head,
                        depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                        use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only,
                        causal_attention=use_causal_attention, relative_position=use_relative_position,
                        temporal_length=temporal_length, is_output_block=False, ds=ds
                    )
                )
            middle_layers.append(
                ResBlock(
                    ch, time_embed_dim, dropout,
                    dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                    use_temporal_conv=temporal_conv
                )
            )
            self.middle_block = TimestepEmbedSequential(*middle_layers)

        self.zero_convolutions = None
        self.input_zero_convolution = None
        if zero_convolution:
            self.input_zero_convolution = ZeroConvolution(in_channels, in_channels)
            self.zero_convolutions = [ZeroConvolution(dim, dim) for dim in input_block_chans]
            if self.middle_block is not None:
                self.zero_convolutions += [ZeroConvolution(ch, ch)]
            self.zero_convolutions = nn.ModuleList(self.zero_convolutions)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
        print(f'Loaded context encoder checkpoint from {checkpoint_path}')

    @torch.no_grad()
    def _prepare_context(self, context, t):
        """
        Handles per-frame image conditioning if present; otherwise repeats text context across frames.
        Expects context shape [b, L, C].
        """
        _, l_context, _ = context.shape
        if l_context == 77 + t * 16:  # NOTE: maintained original hard-coded convention
            context_text, context_img = context[:, :77, :], context[:, 77:, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        return context
    def forward(self, 
            x, 
            condition,
            timesteps,
            mask=None,
            context=None,
            features_adapter=None,
            fs=None,
            **kwargs):
        
        return checkpoint(self._forward, (x, condition, timesteps, mask, context, features_adapter, fs), self.parameters(), self.activation_ckpt)

    def _forward(self, 
                x, 
                condition,
                timesteps,
                mask=None,
                context=None,
                features_adapter=None,
                fs=None,
                **kwargs):
        """
        Encoder forward.

        Inputs:
          - x: [b, C, T, H, W]
          - condition: [b, C, T, H, W]
          - timesteps: [b] or [T] or [b*T] (same usage as original)
          - mask: optional mask tensor used to mask conditioning regions with low confidence [b, 1, T, H, W]
          - context: [b, L, C]; handled per original hard-coded split when L == 77 + T*16
          - features_adapter: optional list of adapter features injected every 3 input blocks
          - fs: optional fps tensor [b] when fs_condition=True

        Returns:
          - h_enc: encoded feature after middle block, shape [(b*T), C, H, W]
          - hs: list of skip features from input blocks (each [(b*T), C, H, W])
          - input_ds: list of downsample rates corresponding to collected features
        """
        b, _, t, _, _ = x.shape
        if self.input_zero_convolution is not None:
            condition = self.input_zero_convolution(condition)
        x = x + condition

        # time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)

        # context handling (repeat per frame, plus optional per-frame image tokens)
        if context is not None:
            context = self._prepare_context(context, t)

        # If timesteps was not per-frame, repeat embedding per-frame
        if timesteps.shape[0] != 16:  # kept as in original
            emb = emb.repeat_interleave(repeats=t, dim=0)

        # combine fps embedding if enabled
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor([self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)
            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        # (b, c, t, h, w) -> (b*t, c, h, w)
        h = rearrange(x, 'b c t h w -> (b t) c h w').type(self.dtype)

        hs = []
        adapter_idx = 0
        for block_id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if block_id == 0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)

            # plug-in adapter features (kept exactly as original: every 3 input blocks)
            if ((block_id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)

        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, 'Wrong features_adapter'

        # middle block
        if self.middle_block is not None:
            h_enc = self.middle_block(h, emb, context=context, batch_size=b)
            hs.append(h_enc)
        # NOTE: keep (b*t, c, h, w) layout to match original decoder expectations.
        if self.zero_convolutions is not None:
            for i, zero_conv in enumerate(self.zero_convolutions):
                hs[i] = rearrange(hs[i], '(b t) c h w -> b c t h w', b=b)
                hs[i] = zero_conv(hs[i])
                hs[i] = rearrange(hs[i], 'b c t h w -> (b t) c h w')
           
            if mask is not None:
                mask = rearrange(mask, 'b 1 t h w -> (b t) 1 h w')
                for i, hs_ in enumerate(hs):
                    hs[i] = hs_ * mask

        return hs, self.input_ds


if __name__ == '__main__':
    from VidUtil.debug import inspect
    from VidUtil.torch_utils import model_summary

    context_encoder_config = {
        "in_channels": 8,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "dropout": 0.1,
        "num_head_channels": 64,
        "transformer_depth": 1,
        "context_dim": 1024,
        "use_linear": True,
        "use_checkpoint": False,
        "temporal_conv": True,
        "temporal_attention": True,
        "temporal_selfatt_only": True,
        "use_relative_position": False,
        "use_causal_attention": False,
        "temporal_length": 16,
        "addition_attention": True,
        "image_cross_attention": True,
        "image_cross_attention_scale_learnable": True,
        "default_fs": 3,
        "fs_condition": True,
        "checkpoint": "../ckpts/dynamicrafter/unet_encoder.pth"
    }
    context_encoder = ContextEncoder(**context_encoder_config).to('cuda')

    model_summary(context_encoder)

    dummy_x = torch.randn(2, 8, 16, 32, 32).to('cuda')
    dummy_t = torch.randint(0, 1000, (2,)).to('cuda')
    dummy_c = torch.randn(2, 77 + 16 * 16, 1024).to('cuda')
    dummy_fs = torch.tensor([3, 6]).to('cuda')
    #dummy_mask = torch.randn(2, 1, 16, 32, 32).to('cuda')

    out = context_encoder(dummy_x, dummy_x, dummy_t, context=dummy_c)

    print('Context encoder output:')
    inspect(out)