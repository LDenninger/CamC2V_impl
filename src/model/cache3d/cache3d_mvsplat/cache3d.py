import os, sys
import torch
import logging
from pathlib import Path
mainlogger = logging.getLogger('mainlogger')

import torch
import numpy as np

try:
    from .mvsplat.src.model.encoder import get_encoder
    from .mvsplat.src.model.decoder import get_decoder
    from .mvsplat.src.config import load_typed_root_config
    from .mvsplat.src.global_cfg import set_cfg
except ImportError as e:
    sys.path.append(os.getcwd())
    sys.path.append(str(Path(os.getcwd())/"src"))
    from src.model.cache3d.cache3d_mvsplat.mvsplat.src.model.encoder import get_encoder
    from src.model.cache3d.cache3d_mvsplat.mvsplat.src.model.decoder import get_decoder
    from src.model.cache3d.cache3d_mvsplat.mvsplat.src.config import load_typed_root_config
    from src.model.cache3d.cache3d_mvsplat.mvsplat.src.global_cfg import set_cfg

from einops import rearrange, repeat

#from .base import Cache3DBase

from omegaconf import OmegaConf
from VidUtil.torch_utils.morphology import Dilation2d, Erosion2d


class MvSplatCache3D:#(Cache3DBase):


    def __init__(self,
                 mvsplat_config: os.PathLike | str = Path(__file__).parent / "default_config.json",
                 height: int = 256,
                 width: int = 256,
                 checkpoint: str =  Path(__file__).parent/"mvsplat"/"checkpoints/re10k.ckpt",
                 device: torch.device = None,
                 sequential: bool = False,
                 offload: bool = False,
                 keep_gaussians_on_device: bool = True,
                 near: float = 1.0,
                 far: float = 100.0,
                 weight_threshold: float = 0.6,
                 weight_dilation_steps: int = 0,
                 weight_erosion_steps: int = 0,
                 weight_dilation_kernel_size: int = 5,
                 weight_erosion_kernel_size: int = 5,
                 debug: bool = False,
                 ):

        self._mvsplat_config_path = mvsplat_config
        self._load_config(mvsplat_config)
        self._offload = offload
        self._keep_gaussians_on_device = keep_gaussians_on_device
        self._sequential = sequential
        self._near = near
        self._far = far
        self._height = height
        self._width = width
        self._device = device if device is not None else torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
        self._debug = debug
        self._weight_dilation_steps = weight_dilation_steps
        self._weight_erosion_steps = weight_erosion_steps
        self._weight_threshold = weight_threshold

        self._gaussians = None

        self._encoder, self._encoder_visualizer = get_encoder(self._config.model.encoder)
        self._encoder = self._encoder.eval()
        self._decoder = get_decoder(self._config.model.decoder, self._config.dataset).eval()

        self._weight_erosion = Erosion2d(kernel_size=weight_erosion_kernel_size).to(self._device)
        self._weight_dilation = Dilation2d(kernel_size=weight_dilation_kernel_size).to(self._device)

        if not self._offload:
            self._encoder = self._encoder.to(self._device)
            self._decoder = self._decoder.to(self._device)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)


    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def update(self, 
                images: torch.Tensor | np.ndarray,
                extrinsics: torch.Tensor | np.ndarray, 
                intrinsics: torch.Tensor | np.ndarray,
                *args, **kwargs):
        """
            Update function to compute the gaussians from input images, extrinsics and intrinsics.

            Args:
                :param images: (B, N, 3, H, W) input images in [0..1] float32
                :param extrinsics: (B, N, 4, 4) input extrinsics
                :param intrinsics: (B, N, 3, 3) input intrinsics. Normalized.
        
        """


        B, N = images.shape[:2]
        near = torch.ones((B, N), device=self._device) * self._near
        far = torch.ones((B, N), device=self._device) * self._far

        if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        if isinstance(extrinsics, np.ndarray): extrinsics = torch.from_numpy(extrinsics)
        if isinstance(intrinsics, np.ndarray): intrinsics = torch.from_numpy(intrinsics)

        images = images.to(self._device)
        extrinsics = extrinsics.to(self._device)
        intrinsics = intrinsics.to(self._device)

        if N % 2 != 0:
            images = torch.cat([images, images[:,0:1]], dim=1)
            extrinsics = torch.cat([extrinsics, extrinsics[:,0:1]], dim=1)
            intrinsics = torch.cat([intrinsics, intrinsics[:,0:1]], dim=1)
            near = torch.cat([near, near[:,0:1]], dim=1)    
            far = torch.cat([far, far[:,0:1]], dim=1)
            N += 1

        if self._offload:
            self.to_device(encoder=True, decoder=False)
        
        if self._sequential:
            gaussians = []
            for b in range(B):
                encoder_input = {
                    "image": images[b:b+1],
                    "extrinsics": extrinsics[b:b+1],
                    "intrinsics": intrinsics[b:b+1],
                    "near": near[b:b+1],
                    "far": far[b:b+1],
                }
                gaussians_b = self._encoder(encoder_input, 0, deterministic=False)
                gaussians.append(gaussians_b)
            
            self._gaussians = gaussians
        else:
            encoder_input = {
                "image": images,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "near": near,
                "far": far,
            }
            self._gaussians = self._encoder(encoder_input, 0, deterministic=False)

        if not self._keep_gaussians_on_device:
            self._gaussians_to_device(torch.device('cpu'))
        if self._offload:
            self.offload(encoder=True, decoder=False)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def render(self,
        poses: torch.Tensor | np.ndarray,
        intrinsics: torch.Tensor | np.ndarray,
        height: int | None = None,
        width: int | None = None,
        return_weight: bool = False,
        filter_weight: bool = True,
    ):
        """
            Rendering a video from the Gaussians in 3D Cache along the provided camera trajectory.

            Args:
                :param poses: (B, N, 4, 4) camera extrinsics (c2w)
                :param intrinsics: (B, N, 3, 3) camera intrinsics (ndc)
                :param height: rendering height
                :param width: rendering width
                :param return_weight: whether to return the weight map along with the rendering
            Returns:
                :return: (B, N, 3, H, W) rendered video in [0..1] float32
        
        """
        B, N = poses.shape[:2]
        height = height or self._height
        width = width or self._width

        near = torch.ones((B, N), device=self._device) * self._near
        far = torch.ones((B, N), device=self._device) * self._far

        if not self._keep_gaussians_on_device:
            self._gaussians_to_device(self._device)

        if isinstance(poses, np.ndarray): poses = torch.from_numpy(poses)
        if isinstance(intrinsics, np.ndarray): intrinsics = torch.from_numpy(intrinsics)
        poses = poses.to(self._device)
        intrinsics = intrinsics.to(self._device)

        if self._offload:
            self.to_device(encoder=False, decoder=True)

        if self._sequential:
            rendering = []
            weight = []
            for b in range(B):
                result = self._decoder.forward(
                    self._gaussians[b],
                    poses[b:b+1],
                    intrinsics[b:b+1],
                    near[b:b+1],
                    far[b:b+1],
                    (height, width),
                    depth_mode=None,
                    render_weight=return_weight,
                )
                rendering.append(result.color)
                weight.append(result.weight)
            rendering = torch.cat(rendering, dim=0)
            if return_weight:
                weight = torch.cat(weight, dim=0)
        else:
            result = self._decoder.forward(
                self._gaussians,
                poses,
                intrinsics,
                near,
                far,
                (height, width),
                depth_mode=None,
                render_weight=return_weight,
            )
            rendering = result.color
            weight = result.weight
        if self._offload:
            self.offload(encoder=False, decoder=True)
        if return_weight:
            if filter_weight:
                weight = self._filter_weight(weight)
            return rendering, weight
        return rendering
    
    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def _filter_weight(self, weight: torch.Tensor):
        """
        Turns a weight map into a binary mask, then applies morphology.
        Fixes soft-bias drift so multiple erosions/dilations won't produce all-black masks.

        Expected shapes:
        - (B, N, C, H, W)  OR  (B, C, H, W)
        Uses self._weight_dilation (Dilation2d) and self._weight_erosion (Erosion2d),
        and self._weight_threshold, *_steps as in your original code.
        """
        if weight.ndim == 5:
            B, N, C, H, W = weight.shape
            x = weight.reshape(B * N, C, H, W)
            needs_reshape = True
        elif weight.ndim == 4:
            B, C, H, W = weight.shape
            x = weight
            needs_reshape = False
        else:
            raise ValueError("weight must be (B,N,C,H,W) or (B,C,H,W)")

        # Normalize per-sample-channel to [0,1] to make threshold stable
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)

        # Initial binary mask
        mask = (x >= self._weight_threshold).float()
        # Apply morphology (modules should be created with matching in/out channels)
        if getattr(self, "_weight_dilation_steps", 0) > 0:
            for _ in range(self._weight_dilation_steps):
                mask = self._weight_dilation(mask)
        mask = (mask >= self._weight_threshold).float()
        if getattr(self, "_weight_erosion_steps", 0) > 0:
            for _ in range(self._weight_erosion_steps):
                mask = self._weight_erosion(mask)

        # Re-binarize after soft/hard ops
        mask = (mask >= self._weight_threshold).float()

        if needs_reshape:
            mask = mask.view(B, N, C, H, W)

        return mask
    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_debug_views(self, images: torch.Tensor | np.ndarray):
        if self._gaussians is None:
            raise ValueError("No gaussians available. Please run update() first.")
        if not self._keep_gaussians_on_device:
            self._gaussians_to_device(self._device)

        if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        images = images.to(self._device)
        B, N = images.shape[:2]
        if N % 2 != 0:
            images = torch.cat([images, images[:,0:1]], dim=1)
            N += 1

        if self._sequential:
            views = []
            for i, g in enumerate(self._gaussians):
                view = self._encoder_visualizer.visualize_gaussians(
                    context_images=images[i:i+1],
                    opacities=g.opacities,
                    covariances=g.covariances,
                    colors=g.harmonics[..., 0],
                )
                views.append(view)
            views = torch.cat(views, dim=0)
        else:
            views = self._encoder_visualizer.visualize_gaussians(
                    context_images=images,
                    opacities=g.opacities,
                    covariances=g.covariances,
                    colors=g.harmonics[..., 0],
                )
        if views.ndim == 3:
            views = views.unsqueeze(0)
        return views



    def load_checkpoint(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location='cpu')
        state_dict = ckpt['state_dict']
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        decoder_state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
        self._encoder.load_state_dict(encoder_state_dict)
        self._decoder.load_state_dict(decoder_state_dict)

    def offload(self, encoder: bool = False, decoder: bool = False):
        if encoder:
            self._encoder.to('cpu')
        if decoder:
            self._decoder.to('cpu')
    
    def to_device(self, encoder: bool = False, decoder: bool = False, device: torch.device = None):
        device = device or self._device
        if encoder:
            self._encoder.to(self._device)
        if decoder:
            self._decoder.to(self._device)

    def _gaussians_to_device(self, device: torch.device = None):
        device = device or self._device
        if self._gaussians is not None:
            if not isinstance(self._gaussians, list):
                self._gaussians.means = self._gaussians.means.to(device)
                self._gaussians.covariances = self._gaussians.covariances.to(device)
                self._gaussians.harmonics = self._gaussians.harmonics.to(device)
                self._gaussians.opacities = self._gaussians.opacities.to(device)
            else:
                for g in self._gaussians:
                    g.means = g.means.to(device)
                    g.covariances = g.covariances.to(device)
                    g.harmonics = g.harmonics.to(device)
                    g.opacities = g.opacities.to(device)
        else:
            mainlogger.warning("[MvSplatCache3D] No gaussians to move to device.")

    def _load_config(self, path: os.PathLike | str):
        """ Load the MvSplat config from a file. """
        if path.suffix in ['.yml', '.yaml']:
            import yaml
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            import json
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        config = OmegaConf.create(config)
        self._config = config
        mainlogger.info(f"[MvSplatCache3D] Loaded config from {path}.")

        set_cfg(config)

    def reset(self):
        self._gaussians = None
        torch.cuda.empty_cache()
        
        
    def eval(self):
        self._encoder.eval()
        self._decoder.eval()
    
    def parameters(self):
        return list(self._encoder.parameters()) + list(self._decoder.parameters())

    def named_parameters(self):
        return self._encoder.named_parameters() + self._decoder.named_parameters()
    
def cache3d_forward_batch(cache3d: MvSplatCache3D, batch, return_weight: bool = False, filter_weight: bool = False):

    def _norm_intrinsic(K, H=256, W=256):
        K[..., 0, 0] /= W
        K[..., 1, 1] /= H
        K[..., 0, 2] /= W
        K[..., 1, 2] /= H
        return K
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

    cache3d.update(images, extrinsics_cond, intrinsics_cond)

    ret = cache3d.render(extrinsics, intrinsics, return_weight=return_weight, filter_weight=filter_weight) # [B, F, C, H, W]
    if return_weight:
        rendered_cond_frames, weight = ret
    else:
        rendered_cond_frames = ret
    rendered_cond_frames = torch.minimum(rendered_cond_frames, torch.ones(1).to(rendered_cond_frames.device))*2. - 1.
    cache3d.reset()

    return rendered_cond_frames, weight


if __name__ == "__main__":
    
    from src.data.utils import get_realestate10k
    from VidUtil import Video
    from pytorch_lightning import seed_everything

    seed_everything(42)
    dataset = get_realestate10k("cvg28")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True, collate_fn=dataset.custom_collate_fn)

    MAX_SAMPES = 40
    OUTPUT = Path("./mvsplat_cache3d_debug")
    if os.path.exists(OUTPUT):
        import shutil
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cache3d = MvSplatCache3D(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        sequential = True,
        offload = False,
        keep_gaussians_on_device = True,
        weight_dilation_steps = 1,
        weight_erosion_steps = 3,
        debug = True,
        weight_threshold=0.2
    )

    for i, batch in enumerate(data_loader):
        if i >= MAX_SAMPES:
            break

        video_name = Path(batch['video_path'][0]).stem
        save_path = OUTPUT / f"{video_name}"
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Video: {video_name}")

        rendering, weight = cache3d_forward_batch(cache3d, batch, return_weight=True, filter_weight=False)

        img_ref = ((batch['video'][:,:,0] +1.)/2.).float().unsqueeze(1) # [B, 1, C, H, W]
        images = ((batch['cond_frames']+1.)/2.).float()
        images = torch.cat([img_ref, images], dim=1) # [B, F, C, H, W]

        weight_filtered = cache3d._filter_weight(weight)

        rendering_v = Video.fromArray(rendering[0], "TCHW")
        weight_v = Video.fromArray(weight[0], "TCHW")
        video_v = Video.fromArray(batch['video'][0], "CTHW")
        weight_filtered_v = Video.fromArray(weight_filtered[0], "TCHW")
        images_v = Video.fromArray(images[0], "TCHW")

        video_save = video_v / rendering_v / weight_v / weight_filtered_v

        video_save.grid("horizontal", file=save_path / "video.png")

        video_path = save_path / "renderings"
        video_path.mkdir(parents=True, exist_ok=True)
        rendering_v.save_frames(video_path)

        weight_path = save_path / "weights"
        weight_path.mkdir(parents=True, exist_ok=True)
        weight_filtered_v.save_frames(weight_path)

        images_path = save_path / "input_images"
        images_path.mkdir(parents=True, exist_ok=True)
        images_v.save_frames(images_path)

