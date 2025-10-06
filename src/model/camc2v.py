import torch
import torch.nn as nn
import logging
import numpy as np

from einops import repeat, rearrange
from utils.geometry import normalize_intrinsics
from utils.utils import instantiate_from_config

from model.cache3d.cache3d_mvsplat import MvSplatCache3D
import random

from typing import *

from model.dynamicrafter import DynamiCrafter
mainlogger = logging.getLogger('mainlogger')

class CamC2V(DynamiCrafter):

    def __init__(self,
                 cache3d_config: dict,
                 diffusion_model_trainable: bool = False,
                 diffusion_model_trainable_parameters: List[str] = [],
                 zero_convolution: bool = False,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.cache3d = instantiate_from_config(cache3d_config)
        
        self.zero_convolution = None
        if zero_convolution:
            self.zero_convolution = nn.Conv3d(4, 4, kernel_size=1)
            nn.init.zeros_(self.zero_convolution.weight)
            nn.init.zeros_(self.zero_convolution.bias)
        
        if diffusion_model_trainable:
            for name, param in self.model.diffusion_model.named_parameters():
                param.requires_grad_(True)
        elif len(diffusion_model_trainable_parameters) > 0:
            for name, param in self.model.diffusion_model.named_parameters():
                if name in diffusion_model_trainable_parameters:
                    param.requires_grad_(True)

    @torch.no_grad()
    def generate(self, 
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
                **kwargs,
                 ):
        
        use_ddim = ddim_steps is not None

        z, c, xrec, x, cache3d_rendering = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=False,
            return_fs=False,
            return_cond_frame=False,
            return_original_input=True,
            return_cache3d_rendering=True,
        )
        N = xrec.shape[0]

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

        with self.ema_scope("Plotting"):
            #with Timer("sample_log"):
            samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=uc, x0=z,
                                                    enable_camera_condition=enable_camera_condition, **kwargs)
        x_samples = self.decode_first_stage(samples)

        if plot_denoise_rows:
            denoise_grid = self._get_denoise_row_from_list(z_denoise_row)

        return {
            "video": x,
            "cache3d_rendering": cache3d_rendering,
            "reconstruction": xrec,
            "samples": x_samples,
            "denoise_grid": denoise_grid if plot_denoise_rows else None,
        }
    
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
        sampled_img_num=None,
        enable_camera_condition=True,
        trace_scale_factor=1.0,
        cond_frame_index=None,
        use_fifo=False,
        **kwargs,
    ):
        # Overwrites functions in parent class
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = sampled_img_num or batch[self.first_stage_key].shape[0]
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
        batch_outputs = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
            return_fs=True,
            return_cond_frame=True,
            return_original_input=True,
            return_cache3d_rendering=True,
            cond_frame_index=cond_frame_index,
        )
        
        # Unpack the actual returns from get_batch_input
        z, c, xrec, xc, fs, cond_x, x, cache3d_rendering = batch_outputs

        #import ipdb; ipdb.set_trace()
        N = xrec.shape[0]
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

  

            with self.ema_scope("Plotting"):
                #with Timer("sample_log"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=uc, x0=z,
                                                        enable_camera_condition=enable_camera_condition, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples


            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log
        

    def get_batch_input(
        self,
        batch,
        random_uncond,
        return_first_stage_outputs: bool = False,
        return_original_cond: bool = False,
        return_fs: bool = False,
        return_cond_frame: bool = False,
        return_original_input: bool = False,
        return_cache3d_rendering: bool = False,
        **kwargs,
    ):    
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        #import ipdb; ipdb.set_trace()

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        ## Generate cache3d rendering for conditioning
        #import ipdb; ipdb.set_trace()
        cache3d_rendering = self.get_cache3d_rendering(batch)
        x = torch.cat([x, rearrange(cache3d_rendering, "B T C H W -> B C T H W ")], dim=2)
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)

        #import ipdb; ipdb.set_trace()
        ## Split latent code into x and cach3d rendering
        z, z_cache3d = torch.split(z, [self.model.diffusion_model.temporal_length, self.model.diffusion_model.temporal_length], dim=2)
        
        ## get caption condition
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

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())


        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                if self.zero_convolution is not None:
                    img_cat_cond = z[:,:,cond_frame_index,:,:]
                    img_cat_cond = img_cat_cond.unsqueeze(2)
                    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

                    img_cat_cond_cache3d = z_cache3d
                    img_cat_cond = img_cat_cond + self.zero_convolution(img_cat_cond_cache3d)
                else:
                    img_cat_cond_cache3d = z_cache3d
                    img_cat_cond = img_cat_cond_cache3d

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_cache3d_rendering:
            out.append(cache3d_rendering)
        return out
    
    @torch.autocast(device_type="cuda", enabled=False)
    def get_cache3d_rendering(self, batch):

        if type(self.cache3d) == MvSplatCache3D:
            _cond_frame_avail = 'cond_frames' in batch and batch['cond_frames'] is not None
            if not _cond_frame_avail:
                return repeat(batch['video'][:,:,0], "B C H W -> B C T H W", T=16)  # [B, C, T, H, W]
            
            intrinsics = batch["camera_intrinsics"].float()
            intrinsics = normalize_intrinsics(intrinsics).float()
            extrinsics = np.linalg.inv(batch["RT_np"])

            intrinsics_cond = batch["camera_intrinsics_cond"].float()
            intrinsics_cond = normalize_intrinsics(intrinsics_cond).float()
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
        else:
            raise ValueError(f"Unknown cache3d type: {type(self.cache3d)}")

        return rendered_cond_frames

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer