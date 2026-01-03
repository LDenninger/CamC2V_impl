
import logging
import pdb

import torch
from einops import rearrange, repeat
from functools import partial

from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.openaimodel3d import TimestepBlock
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
import math
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def unet_downwards_embed(self, x, timesteps, context=None, features_adapter=None, fs=None, camera_condition=None, recursive_condition=None, return_only_skip: bool = False, **kwargs):
    b, _, t, _, _ = x.shape
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
    emb = self.time_embed(t_emb)

    ## repeat t times for context [(b t) 77 768] & time embedding
    ## check if we use per-frame image conditioning
    _, l_context, _ = context.shape
    if l_context == 77 + t * 16:  ## !!! HARD CODE here                     # interp_mode
        context_text, context_img = context[:, :77, :], context[:, 77:, :]
        context_text = context_text.repeat_interleave(repeats=t, dim=0)
        context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
        context = torch.cat([context_text, context_img], dim=1)
    else:
        context = context.repeat_interleave(repeats=t, dim=0)           # single cond frame
    emb = emb.repeat_interleave(repeats=t, dim=0)

    ## always in shape (b t) c h w, except for temporal layer
    x = rearrange(x, 'b c t h w -> (b t) c h w')

    ## combine emb
    if self.fs_condition:
        if fs is None:
            fs = torch.tensor(
                [self.default_fs] * b, dtype=torch.long, device=x.device)
        fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

        fs_embed = self.fps_embedding(fs_emb)
        fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
        emb = emb + fs_embed

    h = x.type(self.dtype)
    adapter_idx = 0
    hs = []
    for id, module in enumerate(self.input_blocks):
        ########################################### only change here, add camera_condition input ###########################################
        if camera_condition is not None:
            camera_condition_input = {}
            camera_condition_input['h'] = h.shape[-2]
            camera_condition_input['w'] = h.shape[-1]
            for key, value in camera_condition.items():
                if 'pluker' in key and value is not None:
                    if self.input_ds[id] in self.attention_resolutions:
                        feature_id = int(math.log2(self.input_ds[id]))
                        camera_condition_input[key] = value[feature_id]
                else:
                    camera_condition_input[key] = value
        else:
            camera_condition_input = None
        h = module(h, emb, context=context, batch_size=b, camera_condition=camera_condition_input)
        if id == 0 and self.addition_attention:
            h = self.init_attn(h, emb, context=context, batch_size=b, camera_condition=None)
        ########################################### only change here, add camera_condition input ###########################################
        ## plug-in adapter features
        if ((id + 1) % 3 == 0) and features_adapter is not None:
            h = h + features_adapter[adapter_idx]
            adapter_idx += 1
        hs.append(h)

    return hs


def new_recursive_forward_for_unet(self, x, timesteps, context=None, features_adapter=None, fs=None, camera_condition=None, recursive_condition=None, **kwargs):
    b, _, t, _, _ = x.shape
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
    emb = self.time_embed(t_emb)

    ## repeat t times for context [(b t) 77 768] & time embedding
    ## check if we use per-frame image conditioning
    _, l_context, _ = context.shape
    if l_context == 77 + t * 16:  ## !!! HARD CODE here                     # interp_mode
        context_text, context_img = context[:, :77, :], context[:, 77:, :]
        context_text = context_text.repeat_interleave(repeats=t, dim=0)
        context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
        context = torch.cat([context_text, context_img], dim=1)
    else:
        context = context.repeat_interleave(repeats=t, dim=0)           # single cond frame
    emb = emb.repeat_interleave(repeats=t, dim=0)

    ## always in shape (b t) c h w, except for temporal layer
    x = rearrange(x, 'b c t h w -> (b t) c h w')

    ## combine emb
    if self.fs_condition:
        if fs is None:
            fs = torch.tensor(
                [self.default_fs] * b, dtype=torch.long, device=x.device)
        fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

        fs_embed = self.fps_embedding(fs_emb)
        fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
        emb = emb + fs_embed

    h = x.type(self.dtype)
    adapter_idx = 0
    hs = []
    for id, module in enumerate(self.input_blocks):
        ########################################### only change here, add camera_condition input ###########################################
        if camera_condition is not None:
            camera_condition_input = {}
            camera_condition_input['h'] = h.shape[-2]
            camera_condition_input['w'] = h.shape[-1]
            for key, value in camera_condition.items():
                if 'pluker' in key and value is not None:
                    if self.input_ds[id] in self.attention_resolutions:
                        feature_id = int(math.log2(self.input_ds[id]))
                        camera_condition_input[key] = value[feature_id]
                else:
                    camera_condition_input[key] = value
        else:
            camera_condition_input = None
        h = module(h, emb, context=context, batch_size=b, camera_condition=camera_condition_input)
        if id == 0 and self.addition_attention:
            h = self.init_attn(h, emb, context=context, batch_size=b, camera_condition=None)
        ########################################### only change here, add camera_condition input ###########################################
        ## plug-in adapter features
        if ((id + 1) % 3 == 0) and features_adapter is not None:
            h = h + features_adapter[adapter_idx]
            adapter_idx += 1
        hs.append(h)
    if features_adapter is not None:
        assert len(features_adapter) == adapter_idx, 'Wrong features_adapter'

    ########################################### only change here, add camera_condition input ###########################################
    if camera_condition is not None:
        camera_condition_input = {}
        camera_condition_input['h'] = h.shape[-2]
        camera_condition_input['w'] = h.shape[-1]
        for key, value in camera_condition.items():
            if 'pluker' in key and value is not None:
                camera_condition_input[key] = value[-1]
            else:
                camera_condition_input[key] = value
    else:
        camera_condition_input = None
    h = self.middle_block(h, emb, context=context, batch_size=b, camera_condition=camera_condition_input)
    ########################################### only change here, add camera_condition input ###########################################

    for id, module in enumerate(self.output_blocks):
        #import ipdb; ipdb.set_trace()
        h_skip = hs.pop()
        h_prev = recursive_condition[id] if recursive_condition is not None else h_skip
        h_skip = self.recursive_insert_blocks[id](h_skip, h_prev, batch_size = b)

        h = torch.cat([h, h_skip], dim=1)
        ########################################### only change here, add camera_condition input ###########################################
        if camera_condition is not None:
            camera_condition_input = {}
            camera_condition_input['h'] = h.shape[-2]
            camera_condition_input['w'] = h.shape[-1]
            for key, value in camera_condition.items():
                if 'pluker' in key and value is not None:
                    if self.output_ds[id] in self.attention_resolutions:
                        feature_id = int(math.log2(self.output_ds[id]))
                        camera_condition_input[key] = value[feature_id]
                else:
                    camera_condition_input[key] = value
        else:
            camera_condition_input = None
        h = module(h, emb, context=context, batch_size=b, camera_condition=camera_condition_input)
        ########################################### only change here, add camera_condition input ###########################################
    h = h.type(x.dtype)
    y = self.out(h)

    # reshape back to (b c t h w)
    y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
    return y
