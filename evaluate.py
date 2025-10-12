import os, sys
import torch
import yaml

sys.path.append(os.path.join(os.getcwd(), "src"))
import argparse
import yaml
import time
from termcolor import colored
from pytorch_lightning import seed_everything
from VidUtil.torch_utils import inspect_checkpoint
from pathlib import Path
import math
import torch.distributed as dist
import signal
from einops import rearrange

from src.data.utils import get_realestate10k
from src.demo.camc2v_demo import CamC2VDemo

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--machine", type=str, default="cvg28", help="Name of the machine")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--sample-file", type=Path, default=None, help="Path to the file containing sample indices")
    parser.add_argument("--only-eval", action="store_true", help="Only run evaluation on existing outputs")
    args = parser.parse_args()

    # --- Distributed init (torchrun) ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0

    # Seed per-rank
    seed_everything(args.seed + rank)

    ## Setup experiment and environment
    machine_env_config = "./configs/env/" + args.machine + ".yaml"
    with open(machine_env_config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    exp_config["experiment_directory"] = Path(exp_config["experiment_directory"]) / args.model

    dataset_args = {"frame_stride": 6}
    if args.sample_file is not None:
        dataset_args["meta_list"] = args.sample_file
    dataset = get_realestate10k(args.machine, **dataset_args)

    if args.sample_file is not None:
        print(f"Using sample file: {args.sample_file}, number of samples: {len(dataset)}")

    # Use DistributedSampler if distributed
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
            collate_fn=dataset.custom_collate_fn, sampler=sampler
        )
        sampler.set_epoch(0)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=dataset.custom_collate_fn
        )

    print(f"[rank {rank}] Number of batches: {len(data_loader)}")
    # Compute per-rank num_samples
    if args.num_samples > 0:
        num_batches = int(math.ceil((args.num_samples / args.batch_size) / max(1, world_size)))
    else:
        num_batches = int(len(data_loader))

    runtime = None

    # Model initialization
    demo_model = CamC2VDemo()

    ckpt_file = None
    config_file = Path(exp_config["experiment_directory"]) / "config.yaml"

    ckpt_step = None
    if args.ckpt is not None:
        try:
            ckpt_path = Path(exp_config["experiment_directory"]) / "checkpoints" 
            

            if args.ckpt in ["last", "last.ckpt"]:
                ckpt_file = Path(exp_config["experiment_directory"]) / "checkpoints" / "last.ckpt"
            else:
                ckpt_file = Path(exp_config["experiment_directory"]) / "checkpoints" / f"{args.ckpt}.ckpt"
            ckpt_file = ckpt_file / "checkpoint" / "mp_rank_00_model_states.pt"
        except Exception as e:
            if rank == 0:
                print(colored(f"Error loading model: {e}", "red"))
            # Clean up before exit if distributed
            if distributed:
                dist.barrier()
                dist.destroy_process_group()
            sys.exit(1)

        print(f"Loading model from {ckpt_file}")
        ckpt_info = inspect_checkpoint(ckpt_file, include_model_summary=False, return_dict=True)
        ckpt_step = ckpt_info["metadata"]["global_step"]
        print(f"Checkpoint global step: {ckpt_step}")
    else:
        print(colored("No checkpoint specified, using pretrained weights!", "yellow"))
    
    if args.output is None:
        args.output = Path("./output") / f"{args.model}_step_{ckpt_step}_sample_{Path(args.sample_file).stem}"
        args.output.mkdir(parents=True, exist_ok=True)

    if args.only_eval:
        if rank == 0:
            print(colored("Only running evaluation on existing outputs...", "yellow"))
        demo_model.evaluate(args.output)
        sys.exit(0)

    demo_model.load_model(
        config_file=config_file,
        ckpt_path=ckpt_file,
        width = 256, height = 256
    )
    # Register SIGINT handler to evaluate partial results on rank 0
    def _handle_sigint(signum, frame):
        try:
            if rank == 0:
                print(colored("SIGINT received. Evaluating existing outputs before exit...", "yellow"))
                demo_model.evaluate(args.output)
        except SystemExit:
            # evaluation already exited the process
            raise
        except Exception as e:
            if rank == 0:
                print(colored(f"Evaluation on SIGINT failed: {e}", "red"))
        finally:
            if distributed:
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    # Every rank creates its output dir (for saving its own results)
    args.output.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        save_config = vars(args).copy()
        save_config["ckpt_path"] = str(ckpt_file)
        save_config["experiment_directory"] = exp_config["experiment_directory"]
        with open(args.output / "config.yaml", "w") as f:
            yaml.dump(save_config, f)

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        if rank == 0:
            print(colored(f"Processing batch {i+1}/{num_batches}", "light_cyan"))

        demo_model.generate(
            video = rearrange(batch['video'], "B C T H W -> B T C H W"), # [B, T, C, H, W]
            extrinsics = batch['RT'], # [B, T, 4, 4]
            intrinsics = batch['camera_intrinsics'], # [B, T, 3,
            condition_frames = batch['cond_frames'], # [B, N, C, H, W]
            extrinsics_condition = batch['RT_cond'], # [B, N, 4, 4]
            intrinsics_condition = batch['camera_intrinsics_cond'], # [B, N, 3, 3]
            frame_stride = batch["frame_stride"],
            caption = batch['caption'],
            video_path = batch['video_path'],
            output_dir=args.output
        )

        if rank == 0:
            iter_left = num_batches - (i + 1)
            est_time_left = iter_left * demo_model._avg_runtime if demo_model._avg_runtime is not None else 0
            eta = time.strftime("%H:%M:%S", time.gmtime(est_time_left))
            print(colored(f"Runtime: {demo_model._avg_runtime:.2f} seconds", "light_cyan"))
            print(colored(f"Estimated time left: {eta}", "light_cyan"))

        # Process the batch

    # Clean up distributed
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    # Run evaluation only on rank 0 after the run completes
    if rank == 0:
        demo_model.evaluate(args.output)