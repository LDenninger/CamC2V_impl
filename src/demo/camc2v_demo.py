import torch
from omegaconf import OmegaConf
from einops import rearrange
import os, sys
sys.path.append(os.path.join(os.getcwd(), "src"))
import time
from pathlib import Path
from model.camc2v import CamC2V

from utils.utils import instantiate_from_config
import numpy as np

from VidUtil import Video, hcat, vcat
from VidUtil.metrics import FVD, SSIM, LPIPS, PSNR, MSE
import tabulate

from termcolor import colored

class CamC2VDemo:

    def __init__(self):

        self._model = None
        self._sample_counter = 0
        self._last_runtime = None
        self._avg_runtime = None
        self._avg_runtime_per_video = None

    def generate(self,
                 video: torch.Tensor, # [B, T, C, H, W]
                 extrinsics: torch.Tensor, # [B, T, 4, 4]
                 intrinsics: torch.Tensor, # [B, T, 3, 3]
                 condition_frames: torch.Tensor, # [B, N, C, H, W]
                 extrinsics_condition: torch.Tensor, # [B, N, 4, 4]
                 intrinsics_condition: torch.Tensor, # [B, N, 3, 3]
                 frame_stride: int,
                 caption: str,

                 depth_maps: torch.Tensor = None, # [B, T, 1, H, W]
                 depth_maps_condition: torch.Tensor = None, # [B, N, 1, H, W]
                 extrinsics_depth_maps: torch.Tensor = None, # [B, T, 4, 4]
                 intrinsics_depth_maps: torch.Tensor = None, # [B, T, 3, 3]
                 extrinsics_depth_maps_condition: torch.Tensor = None, # [B, N, 4, 4]
                 intrinsics_depth_maps_condition: torch.Tensor = None, # [B, N,
  
                 video_path: str = [""],
                 negative_prompt: str = None,
                 fps: int = None,
                 seed: int = 42,
                 cfg_scale: float = 3.5,
                 camera_cfg: float = 1.0,
                 trace_scale_factor: float = 1.0, 
                 eta: float = 1.0,
                 steps: int = 25,
                 output_dir: str = None,

                 enable_camera_condition: bool = False,
                 **kwargs
                 ):
        
        self._sample_counter += 1
        B, T, C, H, W = video.shape
        batch = {
            "video": rearrange(video, "B T C H W -> B C T H W").to("cuda"),
            "RT": extrinsics.to("cuda"),
            "RT_np": extrinsics.cpu().numpy(),
            "camera_intrinsics": intrinsics.to("cuda"),
            "cond_frames": condition_frames.to("cuda"),
            "RT_cond": extrinsics_condition.to("cuda"),
            "RT_cond_np": extrinsics_condition.cpu().numpy(),
            "camera_intrinsics_cond": intrinsics_condition.to("cuda"),
            "depth_maps": depth_maps.to("cuda") if depth_maps is not None else None,
            "depth_maps_cond": depth_maps_condition.to("cuda") if depth_maps_condition is not None else None,
            "RT_depth": extrinsics_depth_maps.to("cuda") if extrinsics_depth_maps is not None else None,
            "camera_intrinsics_depth": intrinsics_depth_maps.to("cuda") if intrinsics_depth_maps is not None else None,
            "RT_depth_cond": extrinsics_depth_maps_condition.to("cuda") if extrinsics_depth_maps_condition is not None else None,
            "camera_intrinsics_depth_cond": intrinsics_depth_maps_condition.to("cuda") if intrinsics_depth_maps_condition is not None else None,
            "caption": caption,
            "video_path": video_path,
            "frame_stride": frame_stride.to("cuda"),
        }
        kwargs.update({
            "ddim_steps": steps,
            "ddim_eta": eta,
            "unconditional_guidance_scale": cfg_scale,
            "timestep_spacing": "uniform_trailing",
            "guidance_rescale": 0.7,
            "camera_cfg": camera_cfg,
            "camera_cfg_scheduler": "constant",
            "enable_camera_condition": enable_camera_condition,
            "cond_frame_index": torch.zeros(B, dtype=torch.long),
            "trace_scale_factor": trace_scale_factor,
            "negative_prompt": negative_prompt,
            "sampled_img_num": B
        })
        #import ipdb; ipdb.set_trace()
        start_time = time.perf_counter()
        output = self._model.log_images(batch, **kwargs)
        end_time = time.perf_counter()
        runtime = (end_time - start_time)
        self._last_runtime = runtime
        if self._avg_runtime is None:
            self._avg_runtime = runtime
            self._avg_runtime_per_video = runtime / B
        else:
            self._avg_runtime = (self._avg_runtime * self._sample_counter + runtime) / (self._sample_counter + 1)
            self._avg_runtime_per_video = (self._avg_runtime_per_video * self._sample_counter * B + runtime) / ((self._sample_counter + 1) * B)
        #import ipdb; ipdb.set_trace()
        if output_dir is not None:
            self._save(batch, output, output_dir)

        return rearrange(output["samples"], "B C T H W -> B T C H W")


    def evaluate(self, path: os.PathLike):
        path = Path(path)
        video_dirs = [d for d in path.iterdir() if d.is_dir()]
        print(colored(f"Found {len(video_dirs)} video directories to evaluate.", "light_cyan"))

        ssim_metric = SSIM()
        lpips_metric = LPIPS()
        psnr_metric = PSNR()
        mse_metric = MSE()
        fvd_metric = FVD()

        _has_cache3d = False
        #import ipdb; ipdb.set_trace()

        for video_dir in video_dirs:

            raw_dir = video_dir / "raw"
            generated_path = raw_dir / "generated.mp4"
            ground_truth_path = raw_dir / "ground_truth.mp4"
            if not generated_path.exists() or not ground_truth_path.exists():
                print(colored(f"Skipping {video_dir} as generated.mp4 or ground_truth.mp4 is missing.", "yellow"))
                continue
            if not _has_cache3d:
                cache3d_rendering_path = video_dir / "cache3d_rendering.mp4"
                if cache3d_rendering_path.exists():
                    _has_cache3d = True

            video_generated = Video.fromFile(generated_path)(format="TCHW")
            video_ground_truth = Video.fromFile(ground_truth_path)(format="TCHW")

            ssim_metric(video_generated, video_ground_truth)
            lpips_metric(video_generated, video_ground_truth)
            psnr_metric(video_generated, video_ground_truth)
            mse_metric(video_generated, video_ground_truth)
            fvd_metric(video_generated, video_ground_truth)

        ssim_generated = ssim_metric.result
        lpips_generated = lpips_metric.result
        psnr_generated = psnr_metric.result
        mse_generated = mse_metric.result
        fvd_generated = fvd_metric.result

        if _has_cache3d:
            ssim_metric.reset()
            lpips_metric.reset()
            psnr_metric.reset()
            mse_metric.reset()
            fvd_metric.reset()

            for video_dir in video_dirs:

                raw_dir = video_dir / "raw"
                ground_truth_path = raw_dir / "ground_truth.mp4"
                cache3d_rendering_path = raw_dir / "cache3d_rendering.mp4"

                if not cache3d_rendering_path.exists():
                    continue

                video_cache3d = Video.fromFile(cache3d_rendering_path)()
                video_ground_truth = Video.fromFile(ground_truth_path)()

                ssim_metric(video_cache3d, video_ground_truth)
                lpips_metric(video_cache3d, video_ground_truth)
                psnr_metric(video_cache3d, video_ground_truth)
                mse_metric(video_cache3d, video_ground_truth)
                fvd_metric(video_cache3d, video_ground_truth)
            
            ssim_cache3d = ssim_metric.result
            lpips_cache3d = lpips_metric.result
            psnr_cache3d = psnr_metric.result
            mse_cache3d = mse_metric.result
            fvd_cache3d = fvd_metric.result

            table = [
                ["Metric", "Generated", "Cache3D"],
                ["SSIM", ssim_generated, ssim_cache3d],
                ["LPIPS", lpips_generated, lpips_cache3d],
                ["PSNR", psnr_generated, psnr_cache3d],
                ["MSE", mse_generated, mse_cache3d],
                ["FVD", fvd_generated, fvd_cache3d],
            ]

        else:    
            table = [
                ["Metric", "Generated"],
                ["SSIM", ssim_generated],
                ["LPIPS", lpips_generated],
                ["PSNR", psnr_generated],
                ["MSE", mse_generated],
                ["FVD", fvd_generated],
            ]
        table_str = tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid")
        with open(path / "000_results.txt", "w") as f:
            f.write(table_str)

        print(table_str)
        sys.exit(0)

    def _save(self, batch: dict, output: dict, output_dir: str, encoding: str = "mp4v"):
        #import ipdb; ipdb.set_trace()
        B = batch["video"].shape[0]
        for b in range(B):
        
            video_name = Path(batch['video_path'][b]).stem
            save_dir = Path(output_dir) / video_name
            raw_save_dir = save_dir / "raw"
            condition_save_dir = save_dir / "condition_frames"
            condition_save_dir.mkdir(parents=True, exist_ok=True)
            save_dir.mkdir(parents=True, exist_ok=True)
            raw_save_dir.mkdir(parents=True, exist_ok=True)

            video_gt = Video.fromArray(batch["video"][b], "CTHW", name="Ground Truth")
            video_generated = Video.fromArray(output["samples"][b], "CTHW", name="Generated")
            video_reconst = Video.fromArray(output["reconst"][b], "CTHW", name="Reconstructed")
            video_save = [video_gt, video_reconst]

            if "cache3d_rendering" in output and output["cache3d_rendering"] is not None:
                video_cache3d = Video.fromArray(output["cache3d_rendering"][b], "TCHW", name="Cache3D Rendering")
                video_save.append(video_cache3d)
            video_save.append(video_generated)

            video_save = hcat(video_save)

            frames_condition = torch.cat([batch["video"][b,:,0].unsqueeze(0), batch["cond_frames"][b]])
            frames_condition = Video.fromArray(frames_condition, "TCHW", name="Condition Frames")
            frames_condition.save_frames(condition_save_dir / "condition_frames")

            #frames_condition.grid('horizontal', file=save_dir / "condition_frames.png")
            video_save.save(save_dir / "video.mp4", fps=5)

            video_gt.save(raw_save_dir / "ground_truth.mp4", fps=7, encoding=encoding)
            video_reconst.save(raw_save_dir / "reconstruction.mp4", fps=7, encoding=encoding)
            if "cache3d_rendering" in output and output["cache3d_rendering"] is not None:
                video_cache3d.save(raw_save_dir / "cache3d_rendering.mp4", fps=7, encoding=encoding)
            video_generated.save(raw_save_dir / "generated.mp4", fps=7, encoding=encoding)

            caption = str(batch["caption"][b])
            with open(save_dir / "caption.txt", "w") as f:
                f.write(caption)
            
            camera_dict = {
                "RT": batch["RT_np"][b],
                "RT_cond": batch["RT_cond_np"][b],
                "camera_intrinsics": batch["camera_intrinsics"][b].cpu().numpy(),
                "camera_intrinsics_cond": batch["camera_intrinsics_cond"][b].cpu().numpy(),
            }
            np.savez(save_dir / "camera_params.npz", **camera_dict)
        
    
    def load_model(self, config_file: str, width: int, height: int, ckpt_path: str = None, device:str="cuda"):
        config = OmegaConf.load(config_file)
        config.model.params.perframe_ae = True
        model = instantiate_from_config(config.model)
        if model.rescale_betas_zero_snr:
            model.register_schedule(
                given_betas=model.given_betas,
                beta_schedule=model.beta_schedule,
                timesteps=model.timesteps,
                linear_start=model.linear_start,
                linear_end=model.linear_end,
                cosine_s=model.cosine_s,
            )

        model.eval()
        for n, p in model.named_parameters():
            p.requires_grad = False

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "module" in state_dict:  # deepspeed checkpoint
                state_dict = state_dict["module"]
            elif "state_dict" in state_dict:  # lightning checkpoint
                state_dict = state_dict["state_dict"]
            state_dict = {k.replace("framestride_embed", "fps_embedding"): v for k, v in state_dict.items()}
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"successfully loaded checkpoint {ckpt_path}")
            except Exception as e:
                print(e)
                model.load_state_dict(state_dict, strict=False)
        else:
            pretrained_ckpt = config.model.pretrained_checkpoint
            pl_sd = torch.load(pretrained_ckpt, map_location="cpu", weights_only=True)
            if "state_dict" in pl_sd:
                model.load_state_dict(pl_sd["state_dict"], strict=False)
            else:
                model.load_state_dict(pl_sd, strict=False)
            print(f"successfully loaded pretrained weights from {pretrained_ckpt}")



        #model.uncond_type = "negative_prompt"
        model = model.to(dtype=torch.float32)
        self._model = model.to(device)

    def to(self, device: str):
        self._model = self._model.to(device)
