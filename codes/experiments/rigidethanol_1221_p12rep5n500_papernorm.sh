#!/bin/bash
#SBATCH --job-name re_pn      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition largemem     # Slurm partition to use
#SBATCH --ntasks 16          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-05:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=30000     # Memory limit for each tasks (in MB)
#SBATCH -o myscript_re_pn%j.out    # File to which STDOUT will be written
#SBATCH -e myscript_re_pnl%j.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate manifold_env_april
python rigidethanol_1221_p12rep5n500_papernorm.py > rigidethanol_122120_p12rep5n500_papernorm.out
