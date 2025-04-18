# Solving Jigsaw Puzzles with Vision Transformers


## Environment Setup

```bash
conda env create -f env.yml -n vlr
```

For full reproducibility, you may use `env-full.yml`, 

Make sure you're already logged into W&B, or else it'll attempt an interactive login
inside a headless SLURM job (which obviously won't work).


## Training

Adjust the parameters in the script as needed (e.g. GPU count, python interpreter paths, etc.)
```bash
sbatch sbatch.sh
```
