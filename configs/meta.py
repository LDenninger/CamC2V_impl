import textwrap

ENVIRONMENT_SETUP = {
    "marvin": textwrap.dedent("""\
        source ~/.bashrc
        module load CUDA/12.4.0
        export WANDB_API_KEY=62173533812ade7ad5afe75ab40494d7ab6fc46e
        export GLOG_minloglevel=2
        conda activate cami2v
    """),
    "cvg28": textwrap.dedent("""\
        source ~/.bashrc
        conda activate cami2v
    """),
    "lamarr": "export WANDB_API_KEY=62173533812ade7ad5afe75ab40494d7ab6fc46e",
    "jureca": textwrap.dedent("""\
        source ~/.bashrc
        module load CUDA/12
        export HF_HOME=/p/project1/westai0081/models/huggingface
        export WANDB_API_KEY=62173533812ade7ad5afe75ab40494d7ab6fc46e
        export WANDB_MODE=offline
        export GLOG_minloglevel=2
        conda activate cami2v
    """),
}