import os
import time
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log

from logging import Logger

from pathlib import Path
import json
from omegaconf import OmegaConf
from typing import Literal

from VidUtil import Video, hcat
from VidUtil.debug import inspect
from einops import rearrange

class ImageLogger(Callback):
    def __init__(self,
                    train_batch_frequency,
                    log_train: bool = True,
                    log_val: bool = True,
                    images_per_batch: int = -1,
                    num_batches=4, 
                    num_val_batches=4,
                    clamp=True,
                    rescale=True,
                    log_first_iteration=False,
                    save_dir=None, 
                    to_local=True,
                    to_tensorboard=False,
                    to_wandb=False, 
                    log_images_kwargs=None, 
                    log_all_gpus=True,
                    image_directory: str = 'images',
                    test_directory: str = None,
                    keys_to_log = ['image_condition','gt_video','samples','cache3d_rendering'],
                    save_suffix=''):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = train_batch_frequency
        self.num_batches = num_batches
        self.num_val_batches = num_val_batches
        self.images_per_batch = images_per_batch
        self.log_first_iteration = log_first_iteration
        self.log_all_gpus = log_all_gpus
        self.to_local = to_local
        self.to_tensorboard = to_tensorboard
        self.to_wandb = to_wandb
        self.clamp = clamp
        self.log_train = log_train
        self.log_val = log_val
        self.keys_to_log = keys_to_log
        self.save_dir = Path(save_dir) / image_directory
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self._cur_log_cnt = 0
        self._cur_mode: Literal["train", "val", "test"] = "none"
        self._log_first_mode = log_first_iteration

        self.test_save_dir = test_directory if test_directory is not None else self.save_dir / "test"
        #import ipdb; ipdb.set_trace()
        if self.to_local:
            ## default save dir
            if os.path.exists(self.save_dir):
                mainlogger.warning(f"Save directory {self.save_dir} already exists. Overwriting.")
                #shutil.rmtree(self.save_dir)

            os.makedirs(self.save_dir, exist_ok=True)
            config_file = self.save_dir / "sample_config.json"
            save_dict = OmegaConf.to_container(self.log_images_kwargs, resolve=True)
            with open(config_file, 'w') as f:
                json.dump(save_dict, f, indent=4)

            os.makedirs(self.save_dir / "train", exist_ok=True)
            os.makedirs(self.save_dir / "val", exist_ok=True)
            os.makedirs(self.test_save_dir, exist_ok=True)

            if "use_fifo" in self.log_images_kwargs and self.log_images_kwargs["use_fifo"]:
                assert "fifo_config" in self.log_images_kwargs, "If using FIFO diffusion, provide 'fifo_config' argument to 'log_images(...)'"
                os.makedirs(self.log_images_kwargs["fifo_config"]["output_dir"], exist_ok=True)


    @torch.no_grad()
    def log_batch_imgs(self, trainer, pl_module, batch, batch_idx, step: int=None, split="train"):
        """ generate images, then save and log to tensorboard """

        step = step or pl_module.global_step
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        batch_logs = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        #import ipdb; ipdb.set_trace()
        batch_size = batch['video'].shape[0]

        visualization_batch = {
            "video": [],
            "condition_frames": [],
        }
        #import ipdb; ipdb.set_trace()
        for b in range(batch_size):
            cache3d_rendering = None
            video_gt = Video.fromArray(batch['video'][b], "CTHW", name="Ground Truth")
            condition_frames = rearrange(batch['cond_frames'][b], "T C H W -> C T H W")
            condition_frames = torch.concatenate([batch["video"][b,:,:1], condition_frames], dim=1)
            condition_frames = Video.fromArray(condition_frames, "CTHW")
            #condition_frames = condition_frames.grid(mode='horizontal')
            video_reconst = Video.fromArray(batch_logs['reconst'][b], "CTHW", name="Reconstruction")
            video_generated = Video.fromArray(batch_logs['samples'][b], "CTHW", name="Generation")

            videos = [video_gt, video_reconst]
            if 'cache3d_rendering' in batch_logs:
                cache3d_rendering = Video.fromArray(batch_logs['cache3d_rendering'][b], "TCHW", name="Cache3D Rendering")
                videos += [cache3d_rendering]
            videos += [video_generated]
            video_save = hcat(videos)

            visualization_batch["video"].append(video_save)
            visualization_batch["condition_frames"].append(condition_frames)

        if self.to_local:

            base_save_dir = self.save_dir / split / f"step_{str(pl_module.global_step).zfill(6)}" 
            batch_info = inspect(batch, print_out = False)

            for b in range(batch_size):
                video_name = Path(batch["video_path"][b]).stem
                batch_save_dir = base_save_dir / video_name
                batch_save_dir.mkdir(parents=True, exist_ok=True)

                visualization_batch["condition_frames"][b].grid(mode='horizontal', file=str(batch_save_dir / "condition_frames.png"))
                visualization_batch["video"][b].save(batch_save_dir / "video.mp4", fps=7)
                with open(batch_save_dir / "batch_info.txt", 'w') as f:
                    f.write(batch_info)
        
        if self.to_wandb:
            video = [v(format='TCHW') for v in visualization_batch["video"]]
            pl_module.logger.log_video(
                key=f"{split}/{str(batch_idx).zfill(4)}/video",
                videos=video,
                step=step,
                fps=[7]*batch_size,
                format=["gif"]*batch_size,           # or "mp4"
            )
            condition_frames = [v.grid(mode='horizontal') for v in visualization_batch["condition_frames"]]
            pl_module.logger.log_image(
                key=f"{split}/{str(batch_idx).zfill(4)}/condition_frames",
                images=condition_frames,
                step=step,
            )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        if not self.log_train:
            return
        #import ipdb; ipdb.set_trace()
        if self.log_first_iteration and pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('train'):
                return
            self.log_first_iteration = False
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, step=0, split="train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.log_train:
            return
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('train'):
                return
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if not self.log_val:
            return
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('val'):
                return
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if not self.log_val:
            return
        if pl_module.logdir and (pl_module.global_rank == 0 or self.log_all_gpus):
            if not self._check_batch('test'):
                return
            mainlogger.info(f'[rank {pl_module.global_rank}] Logging batch {batch_idx}')
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="test")

    def _check_batch(self, mode: Literal[Literal['train', 'val', 'test']]):
        """
            Check whether the current training mode: [train, val, test] has changed and reset internal counter.
            Check whether we have reached the maximum number of batches to log for current mode.

            Returns:
                True, if we should log the current batch.
        """
        if mode != self._cur_mode:
            self._cur_log_cnt = 0
        self._cur_log_cnt += 1
        num_batches = self.num_batches if mode == 'train' else self.num_val_batches
        if num_batches >= 0 and self._cur_log_cnt > num_batches:
            return False
        self._cur_mode = mode
        return True

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

def _now():
    # GPU-accurate if CUDA, otherwise perf_counter
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _fmt_eta(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainWatcherCallback(Callback):
    def __init__(
        self,
        logger: Logger,
        interval: int = 10,
        timing_interval: int = 10,       # how many OPTIMIZER STEPS to aggregate before printing
        timing_max_steps: int = 50,       # how many OPTIMIZER STEPS to aggregate before printing
    ):
        super().__init__()
        self.logger = logger
        self.interval = interval
        self.timing_max_steps = timing_max_steps
        self.timing_interval = timing_interval

        # wall-clock helpers
        self._start_time = None
        self._end_time = None
        self._batch_iter_time = None

        # timing points
        self._t_batch_start = None
        self._t_bw_start = None
        self._t_opt_start = None
        self._last_batch_end = None
        self._is_timing = True

        # running averages + counts
        self._timings = {
            "forward": None,
            "backward": None,
            "optim_step": None,
            "data_load": None,
            "total_iteration": None,
        }
        self._counts = {k: 0 for k in self._timings}

        # control printing once
    # ---------- utilities ----------
    def _update_running_avg(self, key: str, value: float):
        if value is None:
            return
        n = self._counts[key]
        avg = self._timings[key]
        if avg is None:
            new_avg = value
        else:
            new_avg = (avg * n + value) / (n + 1)
        self._timings[key] = new_avg
        self._counts[key] = n + 1

    def _maybe_print_final_timings(self, trainer):

        # print once when we have reached timing_max_steps optimizer steps
        if (trainer.global_step % self.interval) == 0 or trainer.global_step >= trainer.max_steps:
            self.logger.info(
                f"[Timing @ step {trainer.global_step}] "
                f"data_load: {self._timings['data_load'] or 0:.6f}s, "
                f"forward: {self._timings['forward'] or 0:.6f}s, "
                f"backward: {self._timings['backward'] or 0:.6f}s, "
                f"optim_step: {self._timings['optim_step'] or 0:.6f}s, "
                f"total_iteration: {self._timings['total_iteration'] or 0:.6f}s"
            )
        if trainer.global_step >= self.timing_max_steps:
            self._is_timing = False

    # ---------- lifecycle ----------
    def on_fit_start(self, trainer, pl_module):
        self.logger.info("Training is starting.")
        self.logger.info("Trainer configuration:")
        self.logger.info(f"Training iterations/epoch: {trainer.num_training_batches}")
        self.logger.info(f"Validation iterations/epoch: {trainer.num_val_batches}")
        self.logger.info(f"Total iterations (max_steps): {trainer.max_steps}")
        self.logger.info(f"Total epochs: {trainer.max_epochs}")
        self.logger.info(f"Number nodes: {trainer.num_nodes}")
        self.logger.info(f"Number devices: {trainer.num_devices}")

    # ---------- batch timings ----------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # data loading time = gap from previous batch end to this start
        t = _now()
        if self._is_timing and self._last_batch_end is not None:
            self._update_running_avg("data_load", t - self._last_batch_end)
        self._t_batch_start = t
        self._start_time = t  # for your ETA

    def on_before_backward(self, trainer, pl_module, loss):
        # forward time = from batch start to before backward
        if self._is_timing:
            t = _now()
            self._update_running_avg("forward", t - self._t_batch_start)
            self._t_bw_start = t

    def on_after_backward(self, trainer, pl_module):
        # backward time
        if self._is_timing:
            t = _now()
            self._update_running_avg("backward", t - self._t_bw_start)
            self._t_bw_start = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # start optimizer step timer
        if self._is_timing:
            self._t_opt_start = _now()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # this hook runs right after optimizer.step(); use it as "opt end"
        if self._is_timing and self._t_opt_start is not None:
            t = _now()
            self._update_running_avg("optim_step", t - self._t_opt_start)
            self._t_opt_start = None

        # check if we should print final timing aggregates
        self._maybe_print_final_timings(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_time = _now()
        duration = self._end_time - self._start_time
        
        # update total iteration timing
        if self._is_timing and self._t_batch_start is not None:
            self._update_running_avg("total_iteration", duration)
        
        if self._batch_iter_time is None:
            self._batch_iter_time = duration
        else:
            # running mean over optimizer steps shown via global_step;
            # for display we still smooth per call here
            gs = max(trainer.global_step, 1)  # avoid div by zero
            self._batch_iter_time = ((gs - 1) * self._batch_iter_time + duration) / gs

        iter_left = max(trainer.max_steps - (trainer.global_step + 1), 0)
        eta_seconds = iter_left * self._batch_iter_time
        eta_str = _fmt_eta(eta_seconds)

        if (trainer.global_step + 1) % self.interval == 0:
            info_str = f"Epoch [{trainer.current_epoch+1}/{trainer.max_epochs}]"
            info_str += f", iteration [{batch_idx+1}/{trainer.num_training_batches}]"
            info_str += f", global step [{trainer.global_step+1}/{trainer.max_steps}]"
            info_str += f" ETA: {eta_str}"
            try:
                loss_val = (
                    outputs["loss"].item()
                    if isinstance(outputs, dict) and "loss" in outputs
                    else float(outputs)
                )
                info_str += f", loss: {loss_val:.6f}"
            except Exception:
                pass
            self.logger.info(info_str)

        self._last_batch_end = self._end_time
