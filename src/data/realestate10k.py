import os
import numpy as np
import json
import logging

import torch
from torch.utils.data.dataloader import default_collate
from concurrent.futures import ThreadPoolExecutor, as_completed

from math import ceil

from torchvision import transforms
from pathlib import Path

import random
import omegaconf
import subprocess

from typing import List, Literal, Union, Tuple
from torch.utils.data import Dataset

from decord import VideoReader, cpu, gpu


from einops import rearrange
mainlogger = logging.getLogger('mainlogger')


from tqdm import tqdm

fdcomp = None
try:
    import fdcomp
except ImportError:
    pass




class RealEstate10K(Dataset):
    """
        RealEstate10K Dataset Class

        Overview:
        -----------
        A PyTorch `Dataset` that loads short video clips together with per-frame
        camera parameters (poses + intrinsics), optional depth + confidence maps,
        and a free-form caption. It supports sampling a temporal clip from a longer
        video stream with a configurable stride, as well as sampling extra
        "conditioning" frames (context frames) outside or around the main clip.

        Typical Use:
        -------------
        - Training video generation / reconstruction models that need:
          • RGB clip (T frames) normalized to [-1, 1]
          • Camera extrinsics (4x4) and intrinsics (3x3) per frame
          • Optional depth/confidence aligned to frames
          • Optional extra conditioning frames and their camera/depth
          • Text caption for the clip

        Args:
        ------
        :param meta_path (str | os.PathLike):
            Path to the directory containing per-video camera files (*.txt).
            Each file is indexed by `video_name` from `meta_list`.
        :param meta_list (str | os.PathLike):
            Path to a txt file with one `video_name` per line (no extension).
            These names are used to locate `.mp4`, camera `.txt`, and optional
            depth/confidence files.
        :param data_dir (str | os.PathLike):
            Path to the directory containing the video files (`{video_name}.mp4`).
        :param caption_file (str | os.PathLike):
            Path to a JSON file mapping `"${video_name}.mp4"` → caption (str).
        :param depth_dir (str | os.PathLike | None):
            Optional directory of depth files (`{video_name}.dep`). If `None`,
            depth-related outputs will be omitted.
        :param depth_camera_dir (str | os.PathLike | None):
            Optional directory of depth camera extrinsic files (`{video_name}.txt`).
            If `None`, depth-camera outputs will be omitted.
        :param confidence_dir (str | os.PathLike | None):
            Optional directory of confidence files (`{video_name}.dep`). If `None`,
            confidence-related outputs will be omitted.
        :param video_length (int):
            Number of frames T to sample for the main video clip. Use `-1` to keep
            all frames (subsampled by stride).
        :param resolution (List[int]):
            Target [H, W] after resizing/cropping. All image-like tensors are
            resized consistently, and intrinsics are adjusted accordingly.
        :param frame_stride (int | List[int]):
            Temporal stride when sampling the main clip. If an integer, that stride
            is used (with dynamic drop if the video is too short). If a two-element
            list `[min, max]`, a random integer stride is drawn in that range.
        :param frame_stride_for_condition (int | None):
            Temporal stride used when stride-correcting potential indices for
            the *conditioning* frames. Defaults to `frame_stride` if `None`.
        :param random_start_frame (bool):
            If `True`, choose a random valid start index for the clip; otherwise
            start at frame 0.
        :param additional_cond_frames (Literal['none','random','last',"random_true","random_offset","random_full"] | int | List[int]):
            Strategy (or explicit indices/int) to pick extra conditioning frames.
            See `_sample_context_indices` for available strategies.
        :param num_additional_cond_frames (List[int] | int):
            Number of conditioning frames to sample. If a two-element list
            `[min, max]`, a random integer in that range may be used downstream
            (see `custom_collate_fn`); otherwise treated as a fixed count.
        :param suppress_output (bool):
            If `True`, internal `_print` logging is suppressed.
        :param verify (bool):
            If `True`, checks the existence of required files per video and
            prunes `self.video_names` to valid entries.

        __getitem__ Output:
        --------------------
        Returns a dictionary with the following keys (when available):

        Video & Cameras
        - "video": FloatTensor [3, T, H, W], normalized to [-1, 1].
        - "RT": FloatTensor [T, 4, 4], camera extrinsics per main frame (w2c).
        - "camera_intrinsics": FloatTensor [T, 3, 3], per main frame (non-NDC).

        Conditioning (Context) Frames
        - "cond_frames": FloatTensor [N, 3, H, W], extra frames for conditioning.
        - "RT_cond": FloatTensor [N, 4, 4], extrinsics for conditioning frames.
        - "camera_intrinsics_cond": FloatTensor [N, 3, 3], intrinsics for cond frames.

        Text
        - "caption": str, caption for this clip.

        Optional Depth / Confidence (if corresponding dirs provided)
        - "depth_maps": FloatTensor [T, 1, H, W], depth in meters.
        - "depth_maps_cond": FloatTensor [N, 1, H, W], cond depth.
        - "confidence_maps": FloatTensor [T, 1, H, W], logits (unnormalized).
        - "confidence_maps_cond": FloatTensor [N, 1, H, W], cond logits.
        - "RT_depth": FloatTensor [T, 4, 4], depth camera extrinsics (main).
        - "RT_depth_cond": FloatTensor [N, 4, 4], depth camera extrinsics (cond).
        - "camera_intrinsics_depth": FloatTensor [T, 3, 3], depth intrinsics (main).
        - "camera_intrinsics_depth_cond": FloatTensor [N, 3, 3], depth intrinsics (cond).

        Meta
        - "video_path": str, absolute path to the `.mp4` file.
        - "fps": int, effective FPS after applying the final stride.
        - "frame_stride": int, stride used for the main clip.
        - "frame_indices": List[int], indices of the main clip frames.
        - "condition_indices": List[int], indices of the conditioning frames.

        Notes:
        -------
        - Effective FPS is reduced by the chosen stride (integer floor division).
        - If a caption is missing, a different random sample is returned.
        - When optional dirs are `None`, related keys are removed in collate.
    """


    def __init__(self,
                meta_path: os.PathLike,
                meta_list: os.PathLike,
                data_dir: os.PathLike,
                caption_file: os.PathLike,
                depth_dir = None,
                depth_camera_dir = None,
                confidence_dir = None,
                video_length=16,
                resolution=[256, 256],  # H, W
                frame_stride=1,  # [min, max], do not larger than 32 when video_length=16
                frame_stride_for_condition=None,
                random_start_frame=True,
                additional_cond_frames: Literal['none', 'random', 'last', "random_true", "random_offset", "random_full"] | int | List[int]='none',
                num_additional_cond_frames: Union[List[int], int]=0,
                suppress_output=True,
                numpy_keys: List[str] = ["RT", "RT_cond", "RT_depth", "RT_depth_cond"], # keys to keep as NumPy arrays to avoid implicit type casting
                verify: bool = False,
                ):
        super().__init__()
        # --- Data Paths ---
        self.meta_path = Path(meta_path)
        self.meta_list = Path(meta_list)
        self.data_dir = Path(data_dir)
        self.caption_file = Path(caption_file)
        self.depth_dir = Path(depth_dir) if depth_dir is not None else None
        self.depth_camera_dir = Path(depth_camera_dir) if depth_camera_dir is not None else None
        self.confidence_dir = Path(confidence_dir) if confidence_dir is not None else None

        # --- Internal Variables / State ---
        self._none_keys = []            # Keys to drop in collate when features are absent
        self._np_keys = numpy_keys  # Keys to keep as NumPy arrays to avoid implicit type casting
        self._suppress_output = suppress_output
        self.captions = {}              # video_name.mp4 -> caption
        self.video_names = []           # list of base video names (no extension)

        # --- Sampling Configuration ---
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.random_start_frame = random_start_frame
        self.frame_stride_for_condition = frame_stride_for_condition if frame_stride_for_condition is not None else frame_stride
        self.additional_cond_frames = additional_cond_frames
        self.num_additional_cond_frames = num_additional_cond_frames
        if isinstance(self.num_additional_cond_frames, omegaconf.listconfig.ListConfig):
            self.num_additional_cond_frames = list(self.num_additional_cond_frames)
        if isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            self.frame_stride = list(self.frame_stride)

        # Initialize lists/captions and set feature availability flags
        self.initialize()

        # Optional verification pass to filter out missing assets
        if verify:
            valid_video_names, invalid_video_names = self.verify(verbose=False)
            self.video_names = valid_video_names
            mainlogger.info("Verification results:")
            mainlogger.info(f"  Valid videos: {len(valid_video_names)}")
            mainlogger.info(f"  Invalid videos: {len(invalid_video_names)}")
            mainlogger.info(f"  Total videos: {len(self.video_names)}")

        mainlogger.info(f"Dataset initialized with {len(self.video_names)} video clips.")

    ###--- Data Retrieval ---###
    def __getitem__(self, index: int):
        """
        Load one sample consisting of a temporally-subsampled clip plus optional
        conditioning frames and aligned metadata.
        """
        # Wrap-around indexing for safety
        index = index % len(self.video_names)
        video_name = self.video_names[index]

        # --- Resolve per-sample file paths ---
        video_path = self.data_dir / f"{video_name}.mp4"
        camera_path = self.meta_path / f"{video_name}.txt"
        depth_path, confidence_path, depth_camera_path = None, None, None
        if self.depth_dir is not None:
            depth_path = self.depth_dir / f"{video_name}.dep"
        if self.confidence_dir is not None:
            confidence_path = self.confidence_dir / f"{video_name}.dep"
        if self.depth_camera_dir is not None:
            depth_camera_path = self.depth_camera_dir / f"{video_name}.txt"

        # --- Open video and gather basic info ---
        video_reader = VideoReader(str(video_path), ctx=cpu(0))
        video_fps = video_reader.get_avg_fps()
        video_length = len(video_reader)
        if video_length <= self.video_length:
            self._print(f"Video {video_name} has only {video_length} frames (<={self.video_length})", mainlogger.warning)
            return self.__getitem__(random.randint(0, len(self)-1))

        # --- Temporal sampling for main clip (T) ---
        frame_indices, frame_stride = self._sample_frame_indices(video_length) # [T]

        # --- Sample context frame indices (N) according to strategy ---
        context_indices = self._sample_context_indices( # [N]
            strategy = self.additional_cond_frames,
            clip_range = (frame_indices[0], frame_indices[-1]),
            video_length = len(video_reader),
            stride = frame_stride
        )
        # Effective FPS after stride
        video_fps = video_fps // max(1, frame_stride)

        # --- Load required frames (main + context) in one batch ---
        req_indices = frame_indices + context_indices
        video = video_reader.get_batch(req_indices).asnumpy()  # [T+N, H, W, 3], uint8
        video = torch.from_numpy(video)
        video = rearrange(video, "T H W C -> C T H W")  # [3, T+N, H, W]

        # --- Load camera parameters for those frames ---
        camera_pose, fx, fy, cx, cy = load_camera_file(camera_path)
        camera_pose = camera_pose[req_indices]  # [T+N, 4, 4]
        fx, fy, cx, cy = fx[req_indices], fy[req_indices], cx[req_indices], cy[req_indices]  # [T+N, 1]

        # --- Load caption (fallback: resample if missing) ---
        caption = self.captions.get(f"{video_name}.mp4", None)
        if caption is None:
            self._print(f"No caption found for video {video_name}", mainlogger.warning)
            return self.__getitem__(random.randint(0, len(self)-1))
        
        # --- Optional: Load depth, confidence, and depth camera extrinsics ---
        depth_cond, confidence_cond, depth_camera_extrinsics_cond, camera_intrinsics_depth, camera_intrinsics_depth_cond = None, None, None, None, None
        depth, confidence, depth_camera_extrinsics, fxd, fyd, cxd, cyd = load_depth_data(
            depth_path = depth_path,
            confidence_path = confidence_path,
            depth_camera_path = depth_camera_path
        )
        if fxd is not None:
            req_indices_c = [min(ind, fxd.shape[0]-1) for ind in req_indices] # Clamp out-of-bound indices

        if depth is not None:
            depth = depth[req_indices]  # [T+N, H, W], float32
            if fxd is not None:
                fxd = fxd[req_indices_c]
                fyd = fyd[req_indices_c]
                cxd = cxd[req_indices_c]
                cyd = cyd[req_indices_c]

            # Resize/crop to target res and adjust intrinsics for depth camera
            depth, camera_intrinsics_depth, *_ = resize_for_rectangle_crop(depth.unsqueeze(1), self.resolution[0], self.resolution[1], fxd, fyd, cxd, cyd)
            # Split into main (T) and cond (N)
            depth, depth_cond = torch.split(depth, [len(frame_indices), len(context_indices)], dim=0)  # [T, H, W], [N, H, W]
            if camera_intrinsics_depth is not None:
                camera_intrinsics_depth, camera_intrinsics_depth_cond = torch.split(camera_intrinsics_depth, [len(frame_indices), len(context_indices)], dim=0)  # [T, 3, 3], [N, 3, 3]
        if confidence is not None:
            confidence = confidence[req_indices]  # [T+N, H, W], float32
            confidence, *_ = resize_for_rectangle_crop(confidence.unsqueeze(1), self.resolution[0], self.resolution[1])
            confidence, confidence_cond = torch.split(confidence, [len(frame_indices), len(context_indices)], dim=0)  # [T, H, W], [N, H, W]
        if depth_camera_extrinsics is not None:
            depth_camera_extrinsics = depth_camera_extrinsics[req_indices_c]  # [T+N, 4, 4]
            depth_camera_extrinsics,  depth_camera_extrinsics_cond = torch.split(depth_camera_extrinsics, [len(frame_indices), len(context_indices)], dim=0)  # [T, 4, 4], [N, 4, 4]

        # --- Spatial transforms for RGB and pinhole intrinsics (main + cond) ---
        video, camera_intrinsics, *_ = resize_for_rectangle_crop(video, self.resolution[0], self.resolution[1], fx, fy, cx, cy)
        video = video.float() / 255.0  # [3, T+N, H, W] in [0,1]
        video = video*2. - 1. # to [-1, 1]

        # --- Split tensors into main clip (T) and conditioning (N) parts ---
        video, condition_frames = torch.split(video, [len(frame_indices), len(context_indices)], dim=1)
        camera_intrinsics, camera_intrinsics_cond = torch.split(camera_intrinsics, [len(frame_indices), len(context_indices)], dim=0)
        camera_pose, camera_pose_cond = torch.split(camera_pose, [len(frame_indices), len(context_indices)], dim=0)
        condition_frames = rearrange(condition_frames, "C N H W -> N C H W")
        
        # --- Package final sample dictionary ---
        return {
            # Video
            "video": video,  # [3, T, H, W], float32, [-1, 1]
            "RT": camera_pose,  # [T, 4, 4]
            #"RT_np": camera_pose.numpy(),  # [T, 4, 4] # NumPy version to avoid implicit type casting
            "camera_intrinsics": camera_intrinsics,  # [T, 3, 3], non-NDC
            # Context Condition
            "cond_frames": condition_frames,  # [N, 3, H, W]
            "RT_cond": camera_pose_cond,  # [N, 4, 4]
            #"RT_cond_np": camera_pose_cond.numpy(),  # [N, 4, 4] # NumPy version to avoid implicit type casting
            "camera_intrinsics_cond": camera_intrinsics_cond,  # [N, 3, 3], non-NDC
            # Text Condition
            "caption": caption, # str
            # Additional Depth data
            "depth_maps": depth, # [T, 1, H, W], float32 [m]
            "depth_maps_cond": depth_cond,  # [N, 1, H, W], float32 [m]
            "confidence_maps": confidence, # [T, 1, H, W], float32 (unnormalized logits)
            "confidence_maps_cond": confidence_cond, # [N, 1, H, W], float32 (unnormalized logits)
            "RT_depth": depth_camera_extrinsics, # [T, 4, 4]
            "RT_depth_cond": depth_camera_extrinsics_cond, # [N, 4, 4]
            "camera_intrinsics_depth": camera_intrinsics_depth, # [T, 3, 3], non-NDC
            "camera_intrinsics_depth_cond": camera_intrinsics_depth_cond, # [N, 3, 3], non-NDC
            # Meta
            "video_path": str(video_path), # str
            "fps": video_fps, # int
            "frame_stride": frame_stride, # int
            "frame_indices": frame_indices, # List[int]
            "condition_indices": context_indices, # List[int]
        }

    ###--- Utility functions ---###
    def _sample_frame_indices(self, num_frames: int) -> List[int]:
        """
            Dynamically compute the allowed frame strides and randomly sample from it.
            Then define starting frame and sample frame indices accordingly.

            Args:
            :param num_frames (int): Total number of frames in the video clip
            Returns:
            :return frame_indices (List[int]): List of sampled frame indices
        """
        # Note: this routine supports both fixed and random stride. It also
        # attempts to adapt if the source video is shorter than the requested
        # `video_length` at the initial stride.
        frame_stride_drop = 0
        while True:
            if isinstance(self.frame_stride, int):
                # Potential adaptive drop of stride if too few frames are available
                frame_stride = max(self.frame_stride - frame_stride_drop, 1)
            elif (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)) and len(self.frame_stride) == 2:  # [min, max]
                assert (self.frame_stride[0] <= self.frame_stride[1]), f"frame_stride[0]({self.frame_stride[0]}) > frame_stride[1]({self.frame_stride[1]})"
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1])
            else:
                print(type(self.frame_stride))
                print(len(self.frame_stride))
                print(f"frame_stride={self.frame_stride}")
                raise NotImplementedError

            required_frame_num = frame_stride * (self.video_length - 1) + 1
            if num_frames < required_frame_num:
                # If still too short, optionally increase drop or recompute stride
                if isinstance(self.frame_stride, int) and num_frames < required_frame_num * 0.5:
                    frame_stride_drop += 1
                    continue
                else:
                    frame_stride = num_frames // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            break

        # Pick start index and build the index list
        if self.video_length != -1:
            if self.random_start_frame:
                random_range = num_frames - required_frame_num
                start_idx = random.randint(0, random_range) if random_range > 0 else 0
            else:
                start_idx = 0
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
        else:
            frame_indices = list(range(num_frames))
            frame_indices = frame_indices[::frame_stride]

        return frame_indices, frame_stride
    
    def _sample_context_indices(self, 
                                strategy: str,
                                stride: int,
                                clip_range: Tuple[int,int], 
                                video_length: int,
                                offset: int = 0,
                                stride_corrected: bool = False):
        """
        Compute indices for additional conditioning frames (N) according to a strategy.

        Notes on strategies (non-exhaustive):
        - "random_full": sample from the whole video
        - "random_outside": sample from outside [clip_start-offset, clip_end+offset]
        - "random_back": sample after the main clip
        - "random_front": sample before the main clip
        - "last": take the last frame of the clip (+offset)
        - "furthest_distance": pick 0 or last frame of the full video, whichever is further
        - "reference": pick the first frame of the main clip
        - list / int: directly specify indices
        """
        if isinstance(self.num_additional_cond_frames, list) or isinstance(self.num_additional_cond_frames, tuple):
            num_add_cond_frames = self.num_additional_cond_frames[-1]
        else:
            num_add_cond_frames = self.num_additional_cond_frames
        #import ipdb; ipdb.set_trace()
        if isinstance(strategy, list):
            return np.array(strategy)
        elif isinstance(strategy, int):
            return np.array([strategy])

        if strategy == "random_full":
            potential_indices = np.arange(video_length)
        elif strategy == "random_outside":
            potential_indices_front = np.arange(clip_range[0]-offset)
            potential_indices_back = np.arange(clip_range[1]+offset, video_length)
            potential_indices = np.concatenate([potential_indices_front, potential_indices_back])
        elif strategy == "random_back":
            potential_indices = np.arange(clip_range[-1]+offset, video_length)
        elif strategy == "random_front":
            potential_indices = np.arange(0, clip_range[0]-offset)
        elif strategy == "last":
            stride_corrected = False
            potential_indices = np.array([clip_range[-1]+offset])
        elif strategy == "last_2":
            stride_corrected = False
            potential_indices = np.array([min(clip_range[-1]+2, video_length-1)])
        elif strategy == "furthest_distance":
            stride_corrected = False
            dist_front = clip_range[0]
            dist_back = video_length - clip_range[-1]
            potential_indices = np.zeros(1) if dist_front > dist_back else np.ones(1)*(video_length-1)
        elif strategy == "reference":
            stride_corrected = False
            potential_indices = np.array([clip_range[0]])
        else:
            mainlogger.error(f"Unknown strategy '{strategy}' to sample context frames.")
            return []
        if stride_corrected == True and stride >= 0:
            potential_indices = potential_indices[::stride]
        
        num_add_cond_frames = min(potential_indices.shape[0], num_add_cond_frames)
        np.random.shuffle(potential_indices)
        context_indices = potential_indices[:num_add_cond_frames]
        return context_indices.tolist()

    def _print(self, msg: str, printf = mainlogger.info):  
        """Lightweight logger wrapper honoring `suppress_output`."""
        if not self._suppress_output:
            printf(msg)


    ###--- Meta Functions ---###
    def initialize(self):
        """Load `video_names` list and captions; set which optional keys are absent."""
        # Load the list of video identifiers (no extension)
        with open(self.meta_list, 'r') as f:
            self.video_names = [line.strip() for line in f.readlines()]

        # Load captions mapping: "<video_name>.mp4" -> caption string
        with open(self.caption_file, 'r') as f:
            self.captions = json.load(f)

        # Flag which keys should be dropped downstream if optional dirs are missing
        self._none_keys = []
        if self.depth_dir is None:
            self._none_keys += ['depth_maps', 'depth_maps_cond', 'RT_depth', 'RT_depth_cond', 'camera_intrinsics_depth', 'camera_intrinsics_depth_cond']
        if self.confidence_dir is None:
            self._none_keys += ['confidence_maps', 'confidence_maps_cond']
        if self.depth_camera_dir is None:
            self._none_keys += ['RT_depth', 'RT_depth_cond', 'camera_intrinsics_depth', 'camera_intrinsics_depth_cond']

    def verify(self, verbose: bool = False, check_video: bool = False, max_workers: int | None = 32):
        """
        Check presence of required assets per video using a thread pool where
        each thread processes a subset (chunk) of the videos and updates tqdm.

        Args:
        :param verbose (bool):
            If `True`, prints a summary report and missing file details.
        :param check_video (bool):
            If `True`, checks that the video has at least `video_length` frames.
            Significantly slows down the verification process.
        :param max_workers (int | None):
            Number of threads to use. If `None`, lets ThreadPoolExecutor pick.
        """
        def check_one(video_name: str):
            video_path = self.data_dir / f"{video_name}.mp4"
            camera_path = self.meta_path / f"{video_name}.txt"
            depth_path = (self.depth_dir / f"{video_name}.dep") if self.depth_dir is not None else None
            confidence_path = (self.confidence_dir / f"{video_name}.dep") if self.confidence_dir is not None else None
            depth_camera_path = (self.depth_camera_dir / f"{video_name}.txt") if self.depth_camera_dir is not None else None

            valid = True
            missing_msgs = []

            if not video_path.exists():
                valid = False
                missing_msgs.append(f"Missing video file: {video_path}")
            elif check_video:
                fcount = video_frame_count(str(video_path))
                if fcount is not None and fcount < self.video_length:
                    valid = False
                    missing_msgs.append(f"Video too short ({fcount} frames): {video_path}")
            if not camera_path.exists():
                valid = False
                missing_msgs.append(f"Missing camera file: {camera_path}")
            if depth_path is not None and not depth_path.exists():
                valid = False
                missing_msgs.append(f"Missing depth file: {depth_path}")
            if confidence_path is not None and not confidence_path.exists():
                valid = False
                missing_msgs.append(f"Missing confidence file: {confidence_path}")
            if depth_camera_path is not None and not depth_camera_path.exists():
                valid = False
                missing_msgs.append(f"Missing depth camera file: {depth_camera_path}")
            if f"{video_name}.mp4" not in existing_captions:
                valid = False
                missing_msgs.append(f"Missing caption file: {video_name}")

            return valid, missing_msgs

        def chunkify(seq, n_chunks):
            n = len(seq)
            if n_chunks is None or n_chunks <= 0:
                yield seq
                return
            size = ceil(n / n_chunks)
            for i in range(0, n, size):
                yield seq[i:i+size]

        def worker(chunk, pbar):
            local_valid = []
            local_invalid = []
            local_msgs = []
            for vn in chunk:
                is_valid, msgs = check_one(vn)
                if is_valid:
                    local_valid.append(vn)
                else:
                    local_invalid.append(vn)
                    if msgs:
                        # store for printing later to avoid interleaved stdout
                        local_msgs.append((vn, msgs))
                pbar.update(1)  # tqdm is thread-safe for updates
            return local_valid, local_invalid, local_msgs

        valid_video_names = []
        invalid_video_names = []
        all_msgs = []
        existing_captions = set(self.captions.keys())

        video_list = list(self.video_names)
        total = len(video_list)

        # Decide worker count (default: min(32, os.cpu_count() * 5, total) is a common heuristic,
        # but we’ll let ThreadPoolExecutor pick if None)
        max_workers = max_workers or None

        with tqdm(total=total, desc="Verifying videos") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(worker, chunk, pbar) for chunk in chunkify(video_list, ex._max_workers or 1)]
                for fut in as_completed(futures):
                    v, iv, msgs = fut.result()
                    valid_video_names.extend(v)
                    invalid_video_names.extend(iv)
                    all_msgs.extend(msgs)

        if verbose and all_msgs:
            # Use tqdm.write so we don't disturb the finished bar rendering
            for vn, msgs in all_msgs:
                for msg in msgs:
                    tqdm.write(msg)

        if verbose:
            print("Dataset Verification Report:")
            print(f"Total videos checked:... {len(self.video_names)}")
            print(f"Valid videos:........... {len(valid_video_names)}")
            print(f"Invalid videos:......... {len(invalid_video_names)}")

        return valid_video_names, invalid_video_names
    
    def get_index_by_name(self, video_name: str) -> int:
        """Get the dataset index of a video by its base name (no extension)."""
        try:
            return self.video_names.index(video_name)
        except ValueError:
            return None

    def custom_collate_fn(self, batch):
        """
        Custom collate function to subsample a consistent number of `cond_frames` for the entire batch,
        while using the default collate function for all other keys.

        Notes:
        - If `num_additional_cond_frames` is a list [min, max], a random integer
          is chosen uniformly for each batch, then samples are truncated to that N.
        - If the chosen N == 0, condition-related keys are removed from samples.
        - For optional features (depth/confidence/depth-camera) that are globally
          absent (dirs None), their keys are deleted before default_collate; then
          placeholders set to `None` are inserted after collate for consistent keys.
        """
        #import ipdb; ipdb.set_trace()
        if self.additional_cond_frames != 'none':
            if isinstance(self.num_additional_cond_frames, list):
                num_cond_frames = random.randint(self.num_additional_cond_frames[0], self.num_additional_cond_frames[1])
            else:
                num_cond_frames = self.num_additional_cond_frames
            #import ipdb; ipdb.set_trace()
            if num_cond_frames > 0:
                min_num_cond_frames = np.min([s['cond_frames'].shape[0] for s in batch])
                num_cond_frames = min(num_cond_frames, min_num_cond_frames)
                for sample in batch:
                    sample['cond_frames'] = sample['cond_frames'][:num_cond_frames]
                    sample['RT_cond'] = sample['RT_cond'][:num_cond_frames]
                    sample['camera_intrinsics_cond'] = sample['camera_intrinsics_cond'][:num_cond_frames]
                    sample['condition_indices'] = sample['condition_indices'][:num_cond_frames]
                    if sample['depth_maps_cond'] is not None:
                        sample['depth_maps_cond'] = sample['depth_maps_cond'][:num_cond_frames]
                    if sample['RT_depth_cond'] is not None:
                        sample['RT_depth_cond'] = sample['RT_depth_cond'][:num_cond_frames]
                    if sample['camera_intrinsics_depth_cond'] is not None:
                        sample['camera_intrinsics_depth_cond'] = sample['camera_intrinsics_depth_cond'][:num_cond_frames]
                    if sample['confidence_maps_cond'] is not None:
                        sample['confidence_maps_cond'] = sample['confidence_maps_cond'][:num_cond_frames]
            else:
                for sample in batch:
                    del sample['cond_frames']
                    del sample['RT_cond']
                    del sample['camera_intrinsics_cond']
                    del sample['condition_indices']
                    del sample['depth_maps_cond']
                    del sample['confidence_maps_cond']
                    del sample['RT_depth_cond']


        # Drop optional keys that are globally unavailable
        for sample in batch:
            for key in self._none_keys:
                if key in sample:
                    del sample[key]
        
        # Default PyTorch collation for the remaining keys
        data = default_collate(batch)
        #import ipdb; ipdb.set_trace()

        # Reinsert placeholders for missing optional branches to keep a stable dict API
        for key in self._none_keys:
            data[key] = None

        for key in self._np_keys:
            if key in data and data[key] is not None:
                data[f"{key}_np"] = data[key].numpy()
        return data

    def __len__(self):
        """Number of available video clips."""
        return len(self.video_names)


#########################################################
############### Data Loading Functions ##################
#########################################################

def load_camera_file(path: os.PathLike):

    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        camera_data = torch.from_numpy(np.loadtxt(lines)).float()
        n = camera_data.shape[0]
        fx, fy, cx, cy = camera_data[:, 1:5].chunk(4, dim=-1)  # [t,4]
        camera_pose_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
        camera_pose_4x4 = torch.cat([camera_pose_3x4, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * n)], dim=1)  # [t, 4, 4]
    except Exception as e:
        print(f"Error loading camera file {path}: {e}")
        return None

    return camera_pose_4x4, fx, fy, cx, cy

def load_depth_data(
        depth_path: os.PathLike = None,
        confidence_path: os.PathLike = None,
        depth_camera_path: os.PathLike = None
    ):
        def _load(path: str):
            if fdcomp is None:
                raise ImportError("fdcomp module not found. Please install fdcomp to load .dep files.")
            if not os.path.exists(path):
                path.suffix = ".npy"
                if not os.path.exists(path):
                    return None
            data = fdcomp.load(path)
            if isinstance(data, tuple):
                data = data[0]
            return data
        
        depth, confidence, camera_extrinsics = None, None, None
        fx, fy, cx, cy = None, None, None, None
        if depth_path is not None:
            depth = _load(depth_path)
            depth  = torch.from_numpy(depth).float() if depth is not None else None  # [T, H, W]
        if confidence_path is not None:
            confidence = _load(confidence_path)
            if confidence is not None:
                confidence = torch.from_numpy(confidence).float()
        if depth_camera_path is not None:
            camera_extrinsics, fx, fy, cx, cy = load_camera_file(depth_camera_path)

        return depth, confidence, camera_extrinsics, fx, fy, cx, cy

#########################################################
################### Transformations #####################
#########################################################

def resize_for_rectangle_crop(frames, H, W, fx=None, fy=None, cx=None, cy=None):
    '''
    :param frames: C,F,H,W
    :param image_size: H,W
    :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
    '''
    ori_H, ori_W = frames.shape[-2:]
    if ori_W / ori_H > W / H:
        frames = transforms.functional.resize(
            frames,
            size=[H, int(ori_W * H / ori_H)],
        )
    else:
        frames = transforms.functional.resize(
            frames,
            size=[int(ori_H * W / ori_W), W],
        )

    resized_H, resized_W = frames.shape[2], frames.shape[3]
    frames = frames.squeeze(0)

    delta_H = resized_H - H
    delta_W = resized_W - W

    top, left = delta_H // 2, delta_W // 2
    frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

    if fx is None or fy is None or cx is None or cy is None:
        return frames, None, resized_H, resized_W
    
    fx = fx * resized_W
    fy = fy * resized_H
    cx = cx * W
    cy = cy * H
    _1, _0 = torch.ones_like(fx), torch.zeros_like(fx)
    camera_intrinsics = torch.hstack([fx, _0, cx, _0, fy, cy, _0, _0, _1]).reshape(-1, 3, 3)  # [F, 3, 3]

    return frames, camera_intrinsics, resized_H, resized_W

def video_frame_count(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # 1) Try nb_frames (fastest; metadata only)
    cmd1 = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "json", str(path)
    ]
    out1 = subprocess.run(cmd1, capture_output=True, text=True, check=False)
    if out1.returncode == 0:
        data = json.loads(out1.stdout or "{}")
        streams = data.get("streams") or []
        if streams and streams[0].get("nb_frames"):
            return int(streams[0]["nb_frames"])

    # 2) Count frames without decoding (still lightweight)
    cmd2 = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "json", str(path)
    ]
    out2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
    if out2.returncode == 0:
        data = json.loads(out2.stdout or "{}")
        streams = data.get("streams") or []
        if streams and streams[0].get("nb_read_frames"):
            return int(streams[0]["nb_read_frames"])

    # 3) Fallback: duration * avg_frame_rate (approximate for VFR)
    cmd3 = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration,avg_frame_rate",
        "-of", "json", str(path)
    ]
    out3 = subprocess.run(cmd3, capture_output=True, text=True, check=False)
    if out3.returncode == 0:
        data = json.loads(out3.stdout or "{}")
        streams = data.get("streams") or []
        if streams:
            dur = float(streams[0].get("duration") or 0.0)
            afr = streams[0].get("avg_frame_rate") or "0/1"
            # avg_frame_rate arrives like "30000/1001"
            fps = float(Fraction(afr))
            if dur > 0 and fps > 0:
                return int(round(dur * fps))

    raise RuntimeError("Could not determine frame count via ffprobe")


_DATA_PATHS = {
    "marvin": {
        "val": {
            "data_dir":  "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_new/video_clips/test",
            "meta_path": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_new/valid_metadata/test",
            "depth_camera_dir": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/valid_metadata/test",
            "depth_dir": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/depth_maps/test",
            "meta_list": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/test_valid_list.txt",
            "caption_file": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/test_captions.json"
        },
        "train":{
            "data_dir":  "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_new/video_clips/train",
            "meta_path": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_new/valid_metadata/train",
            "depth_camera_dir": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/valid_metadata/train",
            "depth_dir": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/depth_maps/test",
            "meta_list": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/train_valid_list.txt",
            "caption_file": "/lustre/mlnvme/data/s6ludenn_hpc-camcontexti2v/data/realestate10k_own/train_captions.json"
        }
    },
    "jureca": {
        "val": {
            "data_dir":  "/p/project1/westai0081/data/realestate10k_new/video_clips/test",
            "meta_path": "/p/project1/westai0081/data/realestate10k_new/valid_metadata/test",
            "depth_dir": "/p/project1/westai0081/data/realestate10k_own/depth_maps/test",
            "confidence_dir": "/p/project1/westai0081/data/realestate10k_own/confidence_maps/test",
            "depth_camera_dir": "/p/project1/westai0081/data/realestate10k_own/valid_metadata/test",
            "meta_list": "/p/project1/westai0081/data/realestate10k_own/test_valid_list.txt",
            "caption_file": "/p/project1/westai0081/data/realestate10k_own/test_captions.json"
        },
        "train":{
            "data_dir":  "/p/project1/westai0081/data/realestate10k_new/video_clips/train",
            "meta_path": "/p/project1/westai0081/data/realestate10k_new/valid_metadata/train",
            "depth_dir": "/p/project1/westai0081/data/realestate10k_own/depth_maps/train",
            "confidence_dir": "/p/project1/westai0081/data/realestate10k_own/confidence_maps/train",
            "depth_camera_dir": "/p/project1/westai0081/data/realestate10k_own/valid_metadata/train",
            "meta_list": "/p/project1/westai0081/data/realestate10k_own/train_valid_list.txt",
            "caption_file": "/p/project1/westai0081/data/realestate10k_own/train_captions.json"
        }
    },
    "cvg28": {
        "val": {
            "data_dir":  "/home/denninge/marvin/data/realestate10k_new/video_clips/test",
            "meta_path": "/home/denninge/marvin/data/realestate10k_new/valid_metadata/test",
            "depth_camera_dir": "/home/denninge/marvin/data/realestate10k_own/valid_metadata/test",
            "depth_dir": "/home/denninge/marvin/data/realestate10k_own/depth_maps/test",
            "confidence_dir": "/home/denninge/marvin/data/realestate10k_own/confidence_maps/test",
            "meta_list": "/home/denninge/marvin/data/realestate10k_own/test_valid_list.txt",
            "caption_file": "/home/denninge/marvin/data/realestate10k_own/test_captions.json"
        },
        "train":{
            "data_dir":  "/home/denninge/marvin/data/realestate10k_new/video_clips/train",
            "meta_path": "/home/denninge/marvin/data/realestate10k_new/valid_metadata/train",
            "depth_camera_dir": "/home/denninge/marvin/data/realestate10k_own/valid_metadata/train",
            "depth_dir": "/home/denninge/marvin/data/realestate10k_own/depth_maps/train",
            "confidence_dir": "/home/denninge/marvin/data/realestate10k_own/confidence_maps/train",
            "meta_list": "/home/denninge/marvin/data/realestate10k_own/train_valid_list.txt",
            "caption_file": "/home/denninge/marvin/data/realestate10k_own/train_captions.json"
        }
    },
    "lamarr": {
        "val": {
            "data_dir":  "/data/denninge/realestate10k/video_clips/test",
            "meta_path": "/data/denninge/realestate10k/valid_metadata/test",
            "meta_list": "/data/denninge/realestate10k/test_valid_list.txt",
            "caption_file": "/data/denninge/realestate10k/test_captions.json"
        },
        "train": {
            "data_dir":  "/data/denninge/realestate10k/video_clips/train",
            "meta_path": "/data/denninge/realestate10k/valid_metadata/train",
            "meta_list": "/data/denninge/realestate10k/train_valid_list.txt",
            "caption_file": "/data/denninge/realestate10k/train_captions.json"
        }
    }
}

def get_dataset(machine: str, split: Literal["train", "val"] = "val", depth: bool = False, **kwargs):

    data_paths = _DATA_PATHS.get(machine, {}).get(split, {})

    dataset_config = {
        "target": "data.realestate10k.RealEstate10K",
        "params": {
            "data_dir": data_paths["data_dir"],
            "meta_path": data_paths["meta_path"],
            "depth_camera_dir": data_paths.get("depth_camera_dir", None) if depth else None,
            "depth_dir": data_paths.get("depth_dir", None) if depth else None,
            "confidence_dir": data_paths.get("confidence_dir", None) if depth else None,
            "meta_list": kwargs.get("meta_list", data_paths["meta_list"]),
            "caption_file": data_paths["caption_file"],
            "video_length": kwargs.get("video_length", 16),
            "frame_stride": kwargs.get("frame_stride", [1, 10]),
            "resolution": kwargs.get("resolution", [256, 256]),
            "additional_cond_frames": kwargs.get("additional_cond_frames", "random_full"),
            "num_additional_cond_frames": kwargs.get("num_additional_cond_frames", [1, 4]),
            "random_start_frame": kwargs.get("random_start_frame", True),
        }
    }

    dataset = instantiate_from_config(dataset_config)
    return dataset


        
if __name__ == "__main__":
    # Test the dataset class
    import logging
    from VidUtil.debug import inspect
    from VidUtil import Video, hcat, vcat
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-speed", action="store_true", default=False, help="Test loading speed")
    parser.add_argument("--verify", action="store_true", default=False, help="Verify dataset integrity")
    args = parser.parse_args()

    output = Path("./output/new_dataset_test")
    B = 2
    MAX_SAMPLES = 1000
    SPLIT = "train"

    logger = logging.getLogger("mainlogger")
    logger.setLevel(logging.DEBUG)  
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Testing RealEstate10KNew Dataset")

    dataset_params = {
        "data_dir":  f"/home/denninge/marvin/data/realestate10k_new/video_clips/{SPLIT}",
        "meta_path": f"/home/denninge/marvin/data/realestate10k_new/valid_metadata/{SPLIT}",
        "depth_camera_dir": f"/home/denninge/marvin/data/realestate10k_own/valid_metadata/{SPLIT}",
        "depth_dir": f"/home/denninge/marvin/data/realestate10k_own/depth_maps/{SPLIT}",
        #"confidence_dir": f"/home/denninge/marvin/data/realestate10k_own/confidence_maps/{SPLIT}",
        "meta_list": f"/home/denninge/marvin/data/realestate10k_own/{SPLIT}_valid_list.txt",
        "caption_file": f"/home/denninge/marvin/data/realestate10k_own/{SPLIT}_captions.json",
        "video_length": 16,
        "frame_stride": [1, 10],
        "resolution": [256, 256],
        "additional_cond_frames": "random_full",
        "suppress_output": False,
        "num_additional_cond_frames": [1, 4],
    }
    dataset = RealEstate10K(**dataset_params)
    if args.test_speed:

        logger.info("Testing loading speed...")
        runtimes = []
        for i in tqdm(range(MAX_SAMPLES)):
            start_time = time.time()
            _ = dataset[i]
            end_time = time.time()
            duration = end_time - start_time
            if i > 0:
                runtimes.append(duration)
        
        avg_time = sum(runtimes) / len(runtimes)
        logger.info(f"Average time per sample: {avg_time:.4f} seconds over {len(runtimes)} samples")

    if args.verify:
        logger.info("Verifying data...")
        valid_videonames, _ = dataset.verify(check_video=True, verbose=True)

        valid_file = output / "valid_videonames.txt"
        with open(valid_file, "w") as f:
            for name in valid_videonames:
                f.write(f"{name}\n")
        logger.info(f"Valid video names written to {valid_file}")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, num_workers=0, drop_last=True, collate_fn=dataset.custom_collate_fn)
    for i, batch in enumerate(data_loader):

        batch_dir = output / f"batch_{i:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_str = inspect(batch, print_out=False)
        with open(batch_dir / "batch.txt", "w") as f:
             f.write(batch_str)
        print(batch_str)
        videos = []
        condition_frames = []
        depths = []
        depths_cond = []
        #import ipdb; ipdb.set_trace()
        for j in range(B):
            video = Video.fromArray(batch['video'][j], "CTHW", name="Ground Truth")
            condition_frame = Video.fromArray(batch['cond_frames'][j], "TCHW", name="Conditioning Frames")
            depth = Video.fromArray(batch['depth_maps'][j, ..., None], "THWC", name="Depth Maps") if batch.get('depth_maps', None) is not None else None
            depth_cond = Video.fromArray(batch['depth_maps_cond'][j, ..., None], "THWC", name="Conditioning Depth Maps") if batch.get('depth_maps_cond', None) is not None else None
            videos.append(video)
            condition_frames.append(condition_frame)
            depths.append(depth) if depth is not None else None
            depths_cond.append(depth_cond) if depth_cond is not None else None  
        if len(videos) == 1:
            video = videos[0]
            condition_frame = condition_frames[0]
            depth = depths[0] if len(depths) > 0 else None
            depth_cond = depths_cond[0] if len(depths_cond) > 0 else None
        else:
            video = hcat(videos)
            condition_frame = hcat(condition_frames)
            depth = hcat(depths) if len(depths) > 0 else None
            depth_cond = hcat(depths_cond) if len(depths_cond) > 0 else None
        video.grid('vertical', file=batch_dir / "video.png")
        condition_frame.grid('vertical', file=batch_dir / "condition_frames.png")
        if depth is not None:
            depth.grid('vertical', file=batch_dir / "depth_maps.png")
        if depth_cond is not None:
            depth_cond.grid('vertical', file=batch_dir / "depth_maps_cond.png")






