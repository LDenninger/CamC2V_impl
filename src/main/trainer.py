import argparse, os, sys, datetime
# Remapping CUDA devices
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", "0")
import shlex
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch
from einops import rearrange
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[0], '../DynDepth-Anything-V2/metric_depth'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import init_workspace, load_checkpoints, setup_logger, cleanup_logging, save_model_summary
import pdb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from lightning.pytorch.profilers import AdvancedProfiler
import torch.nn.functional as F
import torch.nn as nn
from callbacks import TrainWatcherCallback

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")
    parser.add_argument("--load_from_checkpoint", type=str, default="")
    parser.add_argument("--remap-latent-query", action='store_true', default=False, help="remap latent query tokens to 48x48 resolution")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    # parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    # parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")
    parser.add_argument("--cwd", type=str, default=None, help="Working directory")

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))

def launch_command():
    # Detect if it was started with `-m package.module`
    spec = getattr(sys.modules['__main__'], '__spec__', None)
    if spec and spec.name:
        pieces = [sys.executable, "-m", spec.name, *sys.argv[1:]]
    else:
        pieces = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]

    # Pretty-print with proper quoting
    if os.name == "nt":
        return list2cmdline(pieces)     # Windows-safe quoting
    else:
        return shlex.join(pieces)       # POSIX-safe quoting
    
def print_launch_command(env_keys=None):
    """
    Print the command line that launched this Python process, with fallbacks.
    Also prints selected environment variables useful for torchrun/Slurm.

    Priority:
      1) psutil.Process().cmdline()
      2) /proc/self/cmdline (Linux)
      3) Reconstructed: [sys.executable] + sys.argv
    """
    cmd = None

    # 1) Try psutil (most accurate, cross-platform)
    try:
        import psutil  # pip install psutil (optional)
        cmdline = psutil.Process().cmdline()
        if cmdline:
            cmd = " ".join(shlex.quote(x) for x in cmdline)
    except Exception:
        pass

    # 2) Linux: /proc/self/cmdline
    if cmd is None and os.name == "posix" and os.path.exists("/proc/self/cmdline"):
        try:
            with open("/proc/self/cmdline", "rb") as f:
                parts = f.read().split(b"\0")
            parts = [p.decode() for p in parts if p]
            if parts:
                cmd = " ".join(shlex.quote(x) for x in parts)
        except Exception:
            pass

    # 3) Fallback: reconstruct from sys.executable + sys.argv
    if cmd is None:
        cmd = " ".join([shlex.quote(sys.executable)] + [shlex.quote(a) for a in sys.argv])

    print("\n=== Launch command as seen by this process ===")
    print(cmd)

    # Also show useful env vars for debugging distributed launches
    if env_keys is None:
        env_keys = [
            "CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "RANK", "WORLD_SIZE",
            "SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID", "WANDB_API_KEY", "WANDB_MODE"
        ]
    print("\n=== Selected environment variables ===")
    for k in env_keys:
        v = os.environ.get(k, "not defined")
        if v is not None:
            print(f"{k}={v}")
    print("======================================\n")

ds_cfg = {
    "zero_optimization": {"stage": 1},
    "torch_autocast": {
        "enabled": True,
        "dtype": "bfloat16",
    }
}

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    #import ipdb; ipdb.set_trace()
    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    # parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    #import ipdb; ipdb.set_trace()
    if args.cwd is not None:
        os.chdir(args.cwd)
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed + global_rank, workers=True)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())


    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.logdir, config, lightning_config, global_rank)
    logger = setup_logger(loginfo, dist_rank=global_rank)
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__)) 
    #qlogger.info(f"Launch command: {launch_command()}") 
    print_launch_command()

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)
    cleanup_logging()
    
    logger = setup_logger(loginfo, dist_rank=global_rank)
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__)) 
    # print(model)
    ## load checkpoints
    #import ipdb; ipdb.set_trace()
    if args.load_from_checkpoint == "":
        model = load_checkpoints(model, config.model)
    else:
        logger.info('Do not load pretrained model')

    #! Insert this to remap latent query tokens when using 512 resolution
    #import ipdb; ipdb.set_trace()
    if args.remap_latent_query:
        input_dimension = [32,32]
        output_dimension = [64,64] # 384->48 512 ->64
        query_tokens = model.multi_cond_latent_adaptor.latents.data
        query_tokens = torch.reshape(query_tokens, (-1, *input_dimension, query_tokens.shape[-1]))
        query_tokens = rearrange(query_tokens, 'b h w c -> b c h w')
        query_tokens = F.interpolate(query_tokens, size=output_dimension, mode='bilinear', align_corners=True)
        query_tokens = rearrange(query_tokens, 'b c h w -> b h w c')
        query_tokens = rearrange(query_tokens, 'b h w c -> (b h w) c').unsqueeze(0)
        out_dim = output_dimension[0] * output_dimension[1]
        model.multi_cond_latent_adaptor.latents = nn.Parameter(query_tokens)
        logger.info(f"Remapped latent query tokens from {input_dimension} to {output_dimension}.")


    ## Print configuration
    cfg_yaml = OmegaConf.to_yaml(config, sort_keys=False)
    logger.info("Training configuration:\n%s"%(cfg_yaml))

    ## register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

    #import ipdb; ipdb.set_trace()
    if global_rank == 0:
        #import ipdb; ipdb.set_trace()
        model_summary_file = os.path.join(workdir, "logs", "model_summary.txt")
        save_model_summary(model, model_summary_file)
    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr

    #import ipdb; ipdb.set_trace()
    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #import ipdb; ipdb.set_trace()
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    #logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug, name=args.name)
    logger_cfg = lightning_config.logger

    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)

    trainer_watcher = TrainWatcherCallback(logger=logger, interval=10)
    trainer_config['enable_progress_bar'] = False

    profiler = AdvancedProfiler(filename="perf_logs")
    upstream_logger = instantiate_from_config(logger_cfg)
    trainer = Trainer(
        **trainer_config,
        callbacks=[instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]+[trainer_watcher],
        logger=upstream_logger,
        profiler=profiler if args.debug else None
    )
    logger.info(f"Running on {trainer.num_nodes}x{trainer.num_devices} GPUs")

    if args.load_from_checkpoint:
        logger.info(f"Resume checkpoint from {args.load_from_checkpoint}")
        ckpt_path = args.load_from_checkpoint
    else:
        ckpt_path = None
    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #print("Trainable parameters:")
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"Param: {name}, size: {param.numel()}")
    logger.info("***** Running the Loop *****")
    #import ipdb; ipdb.set_trace()
    # model = torch.compile(model)
    #import ipdb; ipdb.set_trace()
    if args.train:
        if "16" in trainer_config.precision and "deepspeed" in trainer_config.strategy:
            with torch.autocast('cuda'):
                trainer.fit(model, data, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, data, ckpt_path=ckpt_path)
    else:
        
        with torch.autocast('cuda'):
            trainer.test(model, data, ckpt_path=ckpt_path)




    # if args.val:
    #     trainer.validate(model, data)
    # if args.test or not trainer.interrupted:
    #     trainer.test(model, data)