from email import generator
import logging
from math import sqrt
from lvdm.models.samplers.ddim import DDIMSampler
from lora_diffusion import inject_trainable_lora_extended, inject_trainable_lora, tune_lora_scale

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Literal
from functools import partial
import re
import time
import copy
import os
from PIL import Image
import numpy as np

from model.camc2v_cameractrl_controlnet.base import CameraControlLVDM
from model.camc2v_cameractrl_controlnet.multicondi2v_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
)
from model.camc2v_cameractrl_controlnet.unet_forward_function import new_forward_for_unet
from model.camc2v_cameractrl_controlnet.epipolar import Epipolar, EpipolarCrossAttention, pix2coord, VisualCrossAttention
from model.cache3d.cache3d_mvsplat import MvSplatCache3D
from utils.utils import instantiate_from_config
from lvdm.common import extract_into_tensor, default

from VidUtil import Video

mainlogger = logging.getLogger('mainlogger')

def _parse_epipolar_config_default(config):
    if config is None:
        return None
    if not hasattr(config, "is_3d_full_attn"):
        config.is_3d_full_attn = False
    if not hasattr(config, "attention_resolution"):
        config.attention_resolution = [8, 4, 2, 1]
    if not hasattr(config, "apply_epipolar_soft_mask"):
        config.apply_epipolar_soft_mask = False
    if not hasattr(config, "soft_mask_temperature"):
        config.soft_mask_temperature = 1.0
    if not hasattr(config, "epipolar_hybrid_attention"):
        config.epipolar_hybrid_attention = False
    if not hasattr(config, "epipolar_hybrid_attention_v2"):
        config.epipolar_hybrid_attention_v2 = False
    if not hasattr(config, "only_self_pixel_on_current_frame"):
        config.only_self_pixel_on_current_frame = False
    if not hasattr(config, "current_frame_as_register_token"):
        config.current_frame_as_register_token = False
    if not hasattr(config, "pluker_add_type"):
        config.pluker_add_type = "add_to_pre_x_only"
    if not hasattr(config, "add_small_perturbation_on_zero_T"):
        config.add_small_perturbation_on_zero_T = False
    return config


class CamContext2Video(CameraControlLVDM):
    """
    
    Parameters:
        add_type (str): Type of to add the plucker embeddings into the UNet.
        multi_cond_strategy (str): Strategy for combining multiple context condition images.
        recursive_strategy (str): Strategy for recursive sampling for long trajectories.
    """

    def __init__(self,
                    add_type="add_into_temporal_attn",
                    epipolar_config=None,
                    visual_cross_attn_config=None,
                    context_encoder_config=None,
                    plucker_proj_trainable: bool = False,
                    epipolar_attn_trainable: bool = False,
                    pose_guided_cond_trainable: bool = False,
                    first_unet_block_trainable: bool = False,
                    context_encoder_trainable: bool = False,
                    diffusion_model_trainable: bool = False,
                    diffusion_model_trainable_param_list: list = [],
                    first_unet_block_freeze_steps: int = 0,
                    zero_convolution: bool = True,
                    use_semantic_branch: bool = True,
                    epipolar_mask_freeze_steps: int = None,
                    use_pose_embedding_in_latent_adaptor: bool = False,
                    inject_trainable_lora_unet : bool = False,
                    ground_truth_debug_mode: bool = False,
                    cache3d: dict = {},
                    lora_config: dict = {},
                    *args, **kwargs):

        super(CamContext2Video, self).__init__(*args, **kwargs)
        self.add_type = add_type

        ##-- Training Parameters --##
        self.plucker_proj_trainable = plucker_proj_trainable
        self.epipolar_attn_trainable = epipolar_attn_trainable
        self.pose_guided_cond_trainable = pose_guided_cond_trainable
        self.first_unet_block_trainable = first_unet_block_trainable
        self.context_encoder_trainable = context_encoder_trainable
        self.first_unet_block_freeze_steps = first_unet_block_freeze_steps
        self.epipolar_mask_freeze_steps = epipolar_mask_freeze_steps
        self.use_pose_embedding_in_latent_adaptor = use_pose_embedding_in_latent_adaptor
        self.use_zero_convolution = zero_convolution
        self.use_semantic_branch = use_semantic_branch
        self.cache3d_config = cache3d
        self.visual_cross_attn_config = visual_cross_attn_config
        self.ground_truth_debug_mode = ground_truth_debug_mode
        
        # Multi conditioning
        self.cache3d = instantiate_from_config(self.cache3d_config) if self.cache3d_config is not None else None
        context_encoder = instantiate_from_config(context_encoder_config) if context_encoder_config is not None else None
        #self.zero_convolution = None
        #self.multi_latent_supervision_loss_scale = multi_latent_supervision_loss_scale
        #self.multi_latent_l2_loss = 0.0

        # Input adapter
        #self.latent_adapter = instantiate_from_config(latent_adapter_config) if latent_adapter_config is not None else None

        #if self.use_zero_convolution:
        #    self.zero_convolution = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        #    nn.init.constant_(self.zero_convolution.weight, 0.0)
        #    nn.init.constant_(self.zero_convolution.bias, 0.0)
        #    self.zero_convolution.requires_grad_(True)

            #self.multi_cond_in_projection = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
            #nn.init.constant_(self.multi_cond_in_projection.weight, 0.0)
            #nn.init.constant_(self.multi_cond_in_projection.bias, 0.0)
            #self.multi_cond_in_projection.requires_grad_(True)

        self.epipolar_config = _parse_epipolar_config_default(epipolar_config)

        bound_method = new_forward_for_unet.__get__(
            self.model.diffusion_model,
            self.model.diffusion_model.__class__
        )
        setattr(self.model.diffusion_model, 'forward', bound_method)

        if diffusion_model_trainable:
            for name, param in self.model.diffusion_model.named_parameters():
                param.requires_grad_(True)
        elif len(diffusion_model_trainable_param_list) > 0:
            diffusion_model_trainable_param_list = kwargs.get("diffusion_model_trainable_param_list")
            for name, param in self.model.diffusion_model.named_parameters():
                if name in diffusion_model_trainable_param_list:
                    param.requires_grad_(True)
        else:
            for name, param in self.model.diffusion_model.named_parameters():
                param.requires_grad_(False)

        for _name, _module in self.model.diffusion_model.named_modules():
            if _module.__class__.__name__ == 'TemporalTransformer':
                bound_method = new_forward_for_TemporalTransformer.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'TimestepEmbedSequential':
                bound_method = new_forward_for_TimestepEmbedSequential.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)

            elif _module.__class__.__name__ == 'BasicTransformerBlock':
                # SpatialTransformer only
                if _module.context_dim is None and _module.attn1.to_k.in_features != self.model.diffusion_model.init_attn[0].proj_in.out_channels:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

                    bound_method = new_forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, 'forward', bound_method)

                    bound_method = new__forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    if self.pose_encoder is not None:
                        pluker_projection = nn.Linear(_module.attn1.to_k.in_features, _module.attn1.to_k.in_features)
                        nn.init.zeros_(list(pluker_projection.parameters())[0])
                        nn.init.zeros_(list(pluker_projection.parameters())[1])
                        pluker_projection.requires_grad_(plucker_proj_trainable )
                        _module.add_module('pluker_projection', pluker_projection)
                        # _module.add_module('norm_pluker1', nn.LayerNorm(_module.attn1.to_k.in_features))
                        # _module.add_module('norm_pluker2', nn.LayerNorm(_module.attn1.to_k.in_features))

                    if self.epipolar_config is not None:
                        epipolar = Epipolar(
                            query_dim=_module.attn1.to_k.in_features,
                            context_dim=_module.attn1.to_k.in_features,
                            heads=_module.attn1.heads,
                            **self.epipolar_config
                        )
                        _module.add_module('epipolar', epipolar)
                        for _, param in epipolar.named_parameters():
                            param.requires_grad_(epipolar_attn_trainable)
         
        for p in context_encoder.parameters():
            p.requires_grad_(False)
        self.model.diffusion_model.context_encoder = context_encoder

        if inject_trainable_lora_unet:
            mainlogger.info("LoRA adaption enabled!")
            include_conv = lora_config.get('include_conv', False)
            lora_scale = lora_config.get('lora_scale', 1.0)
            if include_conv:
                unet_lora_params, train_names = inject_trainable_lora_extended(self.model)
            else:
                unet_lora_params, train_names = inject_trainable_lora(self.model)

            if lora_scale != 1.0:
                tune_lora_scale(self.model, lora_scale)

            mem_sizes = [sum([p.numel() * p.element_size() for p in pgen]) for pgen in unet_lora_params]
            mem_size_total = sum(mem_sizes)/10**6
            mainlogger.info(f"LoRA parameter size: {mem_size_total} Mb")


    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                    return_cond_frame_index=False, return_cond_frame=False, return_original_input=False, rand_cond_frame=None,
                    enable_camera_condition=True, return_camera_data=False, return_video_path=False, return_depth_scale=False,
                    trace_scale_factor=1.0, cond_frame_index=None, return_presampled_timestep=False, return_cache3d_rendering=False, 
                    return_additional_condition=False, **kwargs):
        
        ## Retrieve input data ##
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        T = x.shape[2]
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        cond_frames = None

        ## Define conditioning frame ##
        if cond_frame_index is None:
            cond_frame_index = torch.zeros(batch_size, device=device, dtype=torch.long)
            rand_cond_frame = self.rand_cond_frame if rand_cond_frame is None else rand_cond_frame
            if rand_cond_frame: 
                cond_frame_index = torch.randint(0, self.model.diffusion_model.temporal_length, (batch_size,), device=device)

        ## Compute camera conditioning ##
        depth_scale = torch.ones((batch_size,), device=device)
        if enable_camera_condition:
            cond_index_inp = cond_frame_index
            camera_condition_log, camera_condition_kwargs = self.get_batch_input_camera_condition_process(
                batch, x, cond_index_inp, trace_scale_factor, rand_cond_frame
            )
            if "depth_scale" in camera_condition_log:
                depth_scale = camera_condition_log["depth_scale"]

        ## Compute the context condition frames using the 3D Cache ##
        if not self.ground_truth_debug_mode:
            rendered_cond_frames = self.get_cache3d_rendering(batch)
            rendered_cond_frames = rearrange(rendered_cond_frames, "B T C H W -> B C T H W ")
            #import ipdb; ipdb.set_trace()
            x = torch.cat([x, rendered_cond_frames], dim=2)
            # Primarily for visualization purposes
            cond_frames = super().get_input(batch, 'cond_frames')
            cond_frames = rearrange(cond_frames, "B T C H W -> B T H W C")
            ref_img = rearrange(x[:,:,0:1], 'B C T H W -> B T H W C')
            cond_frames = torch.cat([ref_img, cond_frames], dim=1)
            cond_frames = rearrange(cond_frames, "B T H W C -> B C T H W ")
            rendered_cond_frames = rearrange(rendered_cond_frames, "B C T H W -> B T C H W ")

        ## Encode to latent space ##
        z = self.encode_first_stage(x)

        ## Cut out additional condition from latents ""
        if self.ground_truth_debug_mode:
            z_cond = z[torch.arange(cond_frame_index.shape[0]),:,cond_frame_index].unsqueeze(2)
            z_add = z
            z_inp = torch.cat([z_cond, z_add[:,:,1:]], dim=2)
            img_cat_cond = z_inp
            _, H, W = x.shape[2:]
        else:
            z_cond = z[torch.arange(cond_frame_index.shape[0]),:,cond_frame_index].unsqueeze(2)
            z_add = z[:,:,-T:]
            z_inp = torch.cat([z_cond, z_add[:,:,1:]], dim=2)
            additional_condition = z_add

        #import ipdb; ipdb.set_trace()
        x = x[:,:,:T]
        z = z[:,:,:T]
        img_cat_cond = z_inp



        ## Encode the caption conditiong ##
        cond_input = batch[self.cond_stage_key]
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        if not hasattr(self, "null_prompt"):
            self.null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, self.null_prompt, cond_emb.detach())

        ## Compute the CLIP image embedding ##
        img = x[torch.arange(batch_size, device=device), :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)  ## b l c
        img_emb = self.image_proj_model(img_emb)

        ## Prepare the model inputs ##
        if self.model.conditioning_key in ['hybrid','layer_fusion']:
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            cond["c_concat"] = [img_cat_cond]  # b c t h w 
            cond["c_cond_frame_index"] = cond_frame_index
            cond["origin_z_0"] = z.clone()
            cond["c_mask"] = None # Potential mask from the rendering process
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)]  ## concat in the seq_len dim

        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
        cond.update(camera_condition_kwargs)
        #import ipdb; ipdb.set_trace()

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])


        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            out.append(fs)
        if return_additional_condition:
            out.append(additional_condition)
        if return_cond_frame_index:
            out.append(cond_frame_index)
        if return_cond_frame:
            if cond_frames is None:
                out.append(x[torch.arange(batch_size, device=device), :, cond_frame_index, ...].unsqueeze(2))
            else:
                out.append(cond_frames)
        if return_original_input:
            out.append(x)
        if return_camera_data:
            camera_data = batch.get('camera_data', None)
            out.append(camera_data)

        if return_video_path:
            out.append(batch['video_path'])

        if return_depth_scale:
            out.append(depth_scale)

        if return_cache3d_rendering:
            #import ipdb; ipdb.set_trace()
            if not self.ground_truth_debug_mode:
                out.append(rendered_cond_frames) # B C T H W
            else:
                out.append(None)
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def get_cache3d_rendering(self, batch):
  
        _cond_frame_avail = 'cond_frames' in batch and batch['cond_frames'] is not None
        if not _cond_frame_avail:
            return repeat(batch['video'][:,:,0], "B C H W -> B C T H W", T=16)  # [B, C, T, H, W]
        
        intrinsics = batch["camera_intrinsics"].float()
        intrinsics = _norm_intrinsic(intrinsics).float()
        extrinsics = np.linalg.inv(batch["RT_np"])

        intrinsics_cond = batch["camera_intrinsics_cond"].float()
        intrinsics_cond = _norm_intrinsic(intrinsics_cond).float()
        intrinsics_cond = torch.cat([intrinsics[:,0:1], intrinsics_cond], dim=1).float()

        extrinsics_cond = np.concatenate([extrinsics[:,0:1], np.linalg.inv(batch["RT_cond_np"])], axis=1)

        img_ref = ((batch['video'][:,:,0] +1.)/2.).float().unsqueeze(1) # [B, 1, C, H, W]
        images = ((batch['cond_frames']+1.)/2.).float()
        images = torch.cat([img_ref, images], dim=1) # [B, F, C, H, W]
        self.cache3d.update(images, extrinsics_cond, intrinsics_cond)

        #gaussian_vis = self.cache3d.get_debug_views(images)
        #gaussian_vis = (gaussian_vis - gaussian_vis.min()) / (gaussian_vis.max() - gaussian_vis.min())
        #gaussian_vis = rearrange(gaussian_vis, "B C H W -> B H W C").cpu().numpy()
        #gaussian_vis = (gaussian_vis*255).astype(np.uint8)
        #Image.fromarray(gaussian_vis[0]).save("gaussian_vis.png")

        rendered_cond_frames = self.cache3d.render(extrinsics, intrinsics) # [B, F, C, H, W]
        rendered_cond_frames = torch.minimum(rendered_cond_frames, torch.ones(1).to(rendered_cond_frames.device))*2. - 1.
        self.cache3d.reset()

        return rendered_cond_frames

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_relative_c2w_RT_pairs(self, RT: Tensor):
        '''
        :param RT: B, T, 4 4   c2w relative RT
        :return: relative RT pairs, c2w, (B, T, T, 4, 4)
        given c2w RT, camera system transform from T1 to T2: inverse(RT_2) @ (RT_1)
        '''

        RT_inv = rearrange(RT.inverse(), "b t ... -> b 1 t ...")
        relative_RT_pairs = RT_inv @ rearrange(RT, "b t ... -> b t 1 ...")  # B, T, T, 4, 4

        return relative_RT_pairs  # B,T,T,4,4

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_fundamental_matrix(self, K: Tensor, R: Tensor, t: Tensor) -> Tensor:
        '''
        :param   K: B, 3, 3
        :param   R: B, 3, 3
        :param   t: B, 3, 1
        :return: F: B, 3, 3
        '''
        E = torch.cross(t, R, dim=-2)
        K_inv = torch.inverse(K)
        F = K_inv.transpose(-1, -2) @ E @ K_inv
        return F

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_epipolar_mask(self, F: Tensor, T: int, H: int, W: int, downsample: int, final_rearrange: bool = True):
        """
        modified to take in batch inputs

        Args:
            grid: (H*W, 3)
            F: camera fundamental matrix (B, T1, T2, 3, 3)
            resolution: feature map resolution H * W
            downsample: downsample scale

        return: weight matrix M(HW * HW)
        """
        # B = F.shape[0]
        device = F.device

        y = torch.arange(0, H, dtype=torch.float, device=device)  # 0 .. 128
        x = torch.arange(0, W, dtype=torch.float, device=device)  # 0 .. 84

        y = pix2coord(y, downsample)  # H
        x = pix2coord(x, downsample)  # W

        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # H * W
        # grid_y: 84x128
        # 3 x HW·
        # TODO check whether yx or xy
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=2).view(-1, 3).float()  # H*W, 3

        lines = F @ grid.transpose(-1, -2)  # [B, T1, T2, 3, H*W]
        norm = torch.norm(lines[..., :2, :], dim=-2, keepdim=True)  # [B, T1, T2, 1, H*W]
        # norm = torch.where(
        #     norm == 0.0,
        #     torch.ones_like(norm),
        #     norm
        # )
        lines = lines / norm  # [B, T1, T2, 3, H*W]

        dist = (lines.transpose(-1, -2) @ grid.transpose(-1, -2)).abs()  # [B, T1, T2, H*W, H*W]
        mask = dist < (downsample * sqrt(2) / 2)  # [B, T1, T2, H*W, H*W]
        # switch to 3d full attention if epipolar mask is empty
        if self.epipolar_config.apply_epipolar_soft_mask:
            raise NotImplementedError
            mask = -dist * self.epipolar_config.soft_mask_temperature  # 高斯分布形式的权重

        if self.epipolar_config.epipolar_hybrid_attention:    # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))

        if self.epipolar_config.epipolar_hybrid_attention_v2:  # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=[2,4], keepdim=True).repeat(1,1,T,1,H*W), mask, torch.ones_like(mask))

        if self.epipolar_config.only_self_pixel_on_current_frame:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1)
            mask = mask * (~same_frame)  # Zero out same frame interactions

            # Step 2: Create identity mask for same pixel in the same frame
            identity_hw = torch.eye(T * H * W, device=device, dtype=mask.dtype).reshape(T, H, W, T, H, W)
            identity_hw = rearrange(
                identity_hw,
                'T1 H1 W1 T2 H2 W2 -> 1 T1 T2 (H1 W1) (H2 W2)'
            ).repeat(mask.shape[0], 1, 1, 1, 1)
            mask = torch.where(identity_hw, identity_hw, mask)

        if self.epipolar_config.current_frame_as_register_token:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1).repeat(mask.shape[0], 1, 1, H * W, H * W)
            mask = torch.where(same_frame, same_frame, mask)
        if final_rearrange:
            return rearrange(mask, "B T1 T2 HW1 HW2 -> B (T1 HW1) (T2 HW2)")
        else:
            return mask

    def add_small_perturbation(self, t, epsilon=1e-6):
        zero_mask = (t.abs() < epsilon).all(dim=-2, keepdim=True)  # 检查 T 的 x, y, z 是否都接近 0
        perturbation = torch.randn_like(t) * epsilon  # 生成微小扰动
        t = torch.where(zero_mask, perturbation, t)  # 如果 T 为零，替换为扰动，否则保持原值

        return t
    
    def compute_conditional_epipolar_mask(self, batch, H, W, downsample_factor: int = 8, return_squeezed_tokens: bool = False, cond_frame_indices: Tensor = None):
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            w2c_RT_4x4_cond = super().get_input(batch, 'RT_cond').float()  # b, c, 4, 4
            #import ipdb; ipdb.set_trace()
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            c2w_RT_4x4_cond = w2c_RT_4x4_cond.inverse()  # w2c --> c2w
            if cond_frame_indices is not None:
                c2w_RT_4x4_cond = torch.cat((c2w_RT_4x4[torch.arange(len(cond_frame_indices)), cond_frame_indices].unsqueeze(1), c2w_RT_4x4_cond), dim=1)

            relative_c2w_RT_4x4 = self.get_pairwise_relative_pose(c2w_RT_4x4_cond, c2w_RT_4x4)
            relative_c2w_RT_4x4 = rearrange(relative_c2w_RT_4x4, 'B T C H W -> B C T H W')

            R = relative_c2w_RT_4x4[..., :3, :3]  # b,t,t,3,3
            t = relative_c2w_RT_4x4[..., :3, 3:4]  # b,t,t,3,1

            C = R.shape[2]
            T = R.shape[1]
            K = repeat(camera_intrinsics_3x3, 'B T H W -> B (T C) H W', C=C)
            R = rearrange(R, 'B T C H W -> B (T C) H W')
            t = rearrange(t, 'B T C H W -> B (T C) H W')
            F = self.get_fundamental_matrix(K, R, t)
            F = rearrange(F, 'B (T C) H W -> B T C H W', C=C)
            
            T = c2w_RT_4x4.shape[1]
            epipolar_mask = self.get_epipolar_mask(F, T, H // downsample_factor, W // downsample_factor, downsample_factor, return_squeezed_tokens)

        return epipolar_mask



    def get_batch_input_camera_condition_process(self, batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame, *args, **kwargs):
        return_log = {}
        return_kwargs = {}
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            #import ipdb; ipdb.set_trace()
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

            relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
            relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

            if self.epipolar_config is not None and not self.epipolar_config.is_3d_full_attn:
                relative_c2w_RT_4x4_pairs = self.get_relative_c2w_RT_pairs(relative_c2w_RT_4x4)  # b,t,t,4,4
                R = relative_c2w_RT_4x4_pairs[..., :3, :3]  # b,t,t,3,3
                t = relative_c2w_RT_4x4_pairs[..., :3, 3:4]  # b,t,t,3,1

                if self.epipolar_config.add_small_perturbation_on_zero_T:
                    t = self.add_small_perturbation(t, epsilon=1e-6)

                K = camera_intrinsics_3x3.unsqueeze(1)


                F = self.get_fundamental_matrix(K, R, t)
                #import ipdb; ipdb.set_trace()
                sample_locs_dict = {d: self.get_epipolar_mask(F, T, H // d, W // d, d) for d in [int(8 * ds) for ds in self.epipolar_config.attention_resolution]}
            else:
                sample_locs_dict = None

        if self.pose_encoder is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                pluker_embedding = self.ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # b, 6, t, H, W

            pluker_embedding_features = self.pose_encoder(pluker_embedding)  # bf c h w
            pluker_embedding_features = [rearrange(_, '(b f) c h w -> b c f h w', b=batch_size) for _ in pluker_embedding_features]
        else:
            pluker_embedding_features = None

        return_kwargs["camera_condition"] = {
            "pluker_embedding_features": pluker_embedding_features,
            "sample_locs_dict": sample_locs_dict,
            "cond_frame_index": cond_frame_index,
            "add_type": self.add_type,
        }

        return return_log, return_kwargs

        
    
    def configure_optimizers(self):
        """
            Overwrites setup from base model
        """
        lr = self.learning_rate
        weight_decay = self.weight_decay

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.pose_encoder is not None and self.pose_encoder_trainable:
            params_pose_encoder = [p for p in self.pose_encoder.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_pose_encoder)}] Paramters for pose_encoder.")
            params.extend(params_pose_encoder)

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))
        
        if self.pose_guided_cond_trainable:
            mainlogger.info(f"@Training [{len(list(self.pose_agent_enc.parameters()))}] Paramters for pose-guided conditioner.")
            params.extend(list(self.pose_agent_enc.parameters()))
        
        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        
        if self.context_encoder_trainable:
            for p in self.model.diffusion_model.context_encoder.parameters():
                p.requires_grad = True
            mainlogger.info(f"@Training [{len(list(self.model.diffusion_model.context_encoder.parameters()))}] Parameters for context_encoder.")
            params.extend(self.model.diffusion_model.context_encoder.parameters())

        #if self.multi_cond_in_projection is not None:
        #    params.extend(list(self.multi_cond_in_projection.parameters()))

        #if self.latent_adapter is not None and self.latent_adapter_trainable:
        #    params_latent_adapter = [p for p in self.latent_adapter.parameters() if p.requires_grad == True]
        #    mainlogger.info(f"@Training [{len(params_latent_adapter)}] Paramters for latent_adapter.")
        #    params.extend(params_latent_adapter)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer
    
    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        mask=None,
        sampled_img_num=1,
        enable_camera_condition=True,
        trace_scale_factor=1.0,
        cond_frame_index=None,
        use_fifo=False,
        **kwargs,
    ):
        # Overwrites functions in parent class
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        for key in batch.keys():
            if batch[key] is None:
                continue
            if isinstance(batch[key], str) \
                or isinstance(batch[key], float) \
                or isinstance(batch[key], int):
                continue
            elif isinstance(batch[key], list) and len(batch[key]) < sampled_img_num:
                continue
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

 
        #with Timer("get_batch_input"):
        z, c, xrec, xc, fs, additional_condition, cond_frame_index, cond_x, x, camera_data, video_path, depth_scale, cache3d_rendering = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
            return_fs=True,
            return_cond_frame_index=True,
            return_cond_frame=True,
            rand_cond_frame=False,
            enable_camera_condition=enable_camera_condition,
            return_original_input=True,
            return_camera_data=True,
            return_video_path=True,
            return_depth_scale=True,
            return_cache3d_rendering=True,
            return_additional_condition=True,
            trace_scale_factor=trace_scale_factor,
            cond_frame_index=cond_frame_index,
        )



        N = xrec.shape[0]
        log["depth_scale"] = depth_scale
        log["camera_data"] = camera_data
        log["video_path"] = video_path
        log["gt_video"] = x
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        if cache3d_rendering is not None:
            log["cache3d_rendering"] = cache3d_rendering 

        if 'cond_frames' in batch.keys():
            log["cond_frames"] = batch["cond_frames"]
        xc_with_fs = []
        #import ipdb; ipdb.set_trace()
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)
                elif self.uncond_type == "negative_prompt":
                    prompts = N * [kwargs["negative_prompt"]]
                    uc_prompt = self.get_learned_conditioning(prompts)

                img = torch.zeros_like(xrec[:, :, 0])  ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img)  ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            pre_process_log, pre_process_kwargs = self.log_images_sample_log_pre_process(
                batch, z, x, cond_frame_index, trace_scale_factor, **kwargs
            )
            log.update(pre_process_log)
            kwargs.update(pre_process_kwargs)
            kwargs.update({"additional_condition": additional_condition})

            with self.ema_scope("Plotting"):
                #with Timer("sample_log"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=uc, x0=z,
                                                        enable_camera_condition=enable_camera_condition, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            log.update(self.log_images_sample_log_post_process(x_samples, **pre_process_log))

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, additional_condition, *add_args = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True, return_additional_condition=True,)
        #import ipdb; ipdb.set_trace()
        kwargs.update({"fs": fs.long()})
        kwargs.update({"additional_condition": additional_condition})
        #import ipdb; ipdb.set_trace()
        loss, loss_dict = self(x, c, **kwargs)
        return loss, loss_dict
    
    def forward(self, x, c, **kwargs):
        if 't' in kwargs:
            t = kwargs['t']
            del kwargs['t']
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)
    
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            #import ipdb; ipdb.set_trace()
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == 'l2_log':
            weight = torch.log10(torch.arange(2, pred.shape[2]+2, device=pred.device))
            weight = (weight / torch.sum(weight))
            weight = rearrange(weight, "N -> 1 1 N 1 1")
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
            loss = weight * loss
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        
        return loss
    
    @torch.no_grad()
    def sample_down_emb_unet(self, x, cond, t, noise=None, **kwargs):
        #import ipdb; ipdb.set_trace()
        # From: LatentDiffusion->p_losses(...)
        if self.noise_strength > 0:
            b, c, f, _, _ = x.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x.device)
            noise = default(noise, lambda: torch.randn_like(x) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        # From: LatentDiffusion->apply_model(...)
        c_concat = cond['c_concat']
        c_crossattn = cond['c_crossattn']
        # From: DiffusionWrapper->forward(...)
        xc = torch.cat([x_noisy] + c_concat, dim=1)
        cc = torch.cat(c_crossattn, 1)

        skip_connections = self.model.diffusion_model.downwards_embed(xc, t, context=cc, **cond, **kwargs)
        skip_connections = [s.detach() for s in skip_connections[::-1]]
        return skip_connections


def _norm_intrinsic(K, H=256, W=256):
    K[..., 0, 0] /= W
    K[..., 1, 1] /= H
    K[..., 0, 2] /= W
    K[..., 1, 2] /= H
    return K