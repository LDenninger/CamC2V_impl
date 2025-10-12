import os
import shutil
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from termcolor import colored
import textwrap
import sys
import shlex
import time
import subprocess

from configs.meta import ENVIRONMENT_SETUP

from src.utils.slurm import run_squeue

from VidUtil.torch_utils import inspect_checkpoint

def run():
    ## Load config
    with initialize(version_base=None, config_path="configs"):
        # 0) Collect CLI overrides (e.g., machine=marvin, env=..., etc.)
        cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]

        # 1) Probe config once WITH CLI to get the effective machine/run_name/exp_dir
        probe = compose(config_name="config", overrides=cli_overrides)

        machine  = probe.machine               # respects CLI: machine=marvin
        run_name = probe.run_name
        exp_dir  = probe.env.experiment_directory

        # Helper: check if user already set a key on the CLI
        def not_set_on_cli(key: str) -> bool:
            prefix = key + "="
            return all(not ov.startswith(prefix) for ov in cli_overrides)

        # 2) Programmatic overrides driven by machine, but don't stomp on explicit CLI choices
        generated = []
        if not_set_on_cli("env"):
            generated.append(f"env={machine}")
        if not_set_on_cli("lightning"):
            generated.append(f"lightning={machine}")
        if not_set_on_cli("data"):
            generated.append(f"data=realestate10k_{machine}")
        
        # 3) Your logging paths
        generated += [
            f'lightning.logger.params.name="{run_name}"',
            f'lightning.logger.params.save_dir="{exp_dir}/{run_name}"',
        ]

        # 4) Final compose: put our generated defaults BEFORE CLI so the user can still override them
        cfg = compose(config_name="config", overrides=generated + cli_overrides)

    cfg_yaml = OmegaConf.to_yaml(cfg, sort_keys=False)

    if cfg.debug:
        print(colored("Debug mode is ON.", "yellow"))
        print("Run configuration:")
        print(cfg_yaml)

    ## Setup run directory
    ckpt_path = None
    run_path = Path(cfg.env.experiment_directory) / cfg.run_name
    log_dir = run_path / "logs"
    image_dir = run_path / "images"
    checkpoint_dir = run_path / "checkpoints"
    stdout_file = log_dir / "stdout.txt"

    if cfg.checkpoint is not None:
        if not os.path.exists(run_path):
            print(colored(f"Trying to resume training from non-existing run '{cfg.run_name}' -> {run_path}", "red"))
            return
        ckpt_path = run_path / "checkpoints" / f"{cfg.checkpoint}.ckpt"
        if not os.path.exists(ckpt_path):
            print(colored(f"Checkpoint {ckpt_path} does not exist!", "red"))
            return
        print(colored(f"Resuming training from checkpoint: {ckpt_path}", "green"))
        print(inspect_checkpoint(ckpt_path / "checkpoint" / "mp_rank_00_model_states.pt", include_model_summary=False))

        config_file = run_path / "config.yaml"
        print(f"Using configuration from run: {config_file}")
        cfg = OmegaConf.load(config_file)

        # Re-apply CLI overrides with proper type parsing
        dot_overrides = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, dot_overrides)

        # (optional) re-dump for debugging
        cfg_yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
    else:

        if not os.path.exists(cfg.env.experiment_directory):
            os.makedirs(cfg.env.experiment_directory)
            print(colored(f"Created experiment directory: {cfg.env.experiment_directory}", "green"))
        
        if os.path.exists(run_path):
            if cfg.override or cfg.run_name == "debug" or cfg.debug:
                print(colored(f"Overriding existing run at {run_path}", "yellow"))
                shutil.rmtree(run_path)
                os.makedirs(run_path)
            else:
                print(f"Run directory {run_path} already exists.")
                inp = input("Do you want to override it? (y/n): ")
                if inp.lower() == 'y':
                    shutil.rmtree(run_path)
                    os.makedirs(run_path)
                else:
                    print(colored("Exiting to avoid overwriting.", 'red'))
                    return
        

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    cfg.lightning.logger.params.save_dir = f"{cfg.env.experiment_directory}/{cfg.run_name}"
    cfg.lightning.callbacks.model_checkpoint.params.dirpath = f"{cfg.env.experiment_directory}/{cfg.run_name}/checkpoints"
    cfg_file = run_path / "config.yaml"
    with open(cfg_file, 'w') as f:
        f.write(cfg_yaml)

    print(colored(f"Run directory set up at: {run_path}", "green"))

    ## Save the command to start this script
    run_cmd_file = run_path / "train_cmd.txt"
    with open(run_cmd_file, "w") as f:
        executable = sys.executable
        executable = executable.split("/")[-1]  
        cmd = [executable] + sys.argv
        cmd = shlex.join(cmd)
        f.write(cmd)

    ## Create run script
    run_script = "#!/bin/bash\n"
    run_script_file = str(run_path / "train")
    slurm_cmds, env_setup, run_command = "", "", ""
    env_setup = ENVIRONMENT_SETUP.get(cfg.machine, "")

    if cfg.schedule:
        slurm_cmds = textwrap.dedent(f"""\
            #SBATCH --partition={cfg.env.partition}
            #SBATCH --account {cfg.env.account}
            #SBATCH --job-name={cfg.run_name}
            #SBATCH --output={stdout_file}
            #SBATCH --error={stdout_file}
            #SBATCH --cpus-per-task={cfg.env.cpus_per_task}            
            #SBATCH --ntasks={cfg.lightning.trainer.devices}
            #SBATCH --ntasks-per-node={cfg.lightning.trainer.devices}
            #SBATCH --mem-per-cpu={cfg.env.mem_per_cpu}G             
            #SBATCH --nodes={cfg.lightning.trainer.num_nodes}
            #SBATCH --gpus={cfg.lightning.trainer.devices}
            #SBATCH --time={cfg.env.run_time}   
        """)
        if cfg.machine == "marvin": slurm_cmds += "#SBATCH --export NONE\n"
        run_script_file += ".slurm"
    else:
        run_script_file += ".sh"

    run_command = f"torchrun --standalone --nproc_per_node={cfg.lightning.trainer.devices} --node_rank=0 --rdzv_id=12345 --rdzv_backend=c10d {cfg.env.source_directory}/src/main/trainer.py"
    python_args = {
        "name": cfg.run_name,
        "base": str(cfg_file),
        "logdir": run_path,
    }
    if cfg.checkpoint is not None:
        python_args["load_from_checkpoint"] = str(ckpt_path)
    
    run_command += ' ' + ' '.join([f"--{k} {v}" for k, v in python_args.items()]) + ' --train'

    run_script += f"{slurm_cmds}\n"
    run_script += f"{env_setup}\n"
    run_script += f"{run_command}\n"

    with open(run_script_file, 'w') as f:
        f.write(run_script)

    print("Training configuration:")
    print(f" Run name: {cfg.run_name}")
    print(f" Run path: {run_path}")
    print(f" Stdout: {stdout_file}")
    print(f" Train script: {run_script_file}")
    print(f" Config file: {cfg_file}")
    print(f" Machine: {cfg.machine}")
    print(f" Num nodes: {cfg.lightning.trainer.num_nodes}")
    print(f" Devices per node: {cfg.lightning.trainer.devices}")
    print(f" Slurm: {cfg.schedule}")

    print(colored("\n--- Run command ---", "light_cyan"))
    print(run_command)
    print(colored("-------------------\n", "light_cyan"))

    ## Start training
    if cfg.schedule:
        slurm_command = ["sbatch"]
        if cfg.dependency is not None:
            slurm_command.append(f"--dependency={cfg.dependency}")
        slurm_command.append(str(run_script_file))

        result = subprocess.run(slurm_command, capture_output=True)
        if result.returncode == 0:
            print(colored("Job submitted successfully.", "green"))
            print(result.stdout.decode('utf-8') + "\n")
        else:
            print(colored("Error submitting job:\n", "red"), result.stderr.decode('utf-8'))
            return
        
        time.sleep(2)
        print(run_squeue())
    else:
        print(colored("Starting training script...", "green"))
        os.execv("/bin/bash", ["bash", str(run_script_file)])


if __name__=="__main__":
    run()