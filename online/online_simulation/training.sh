#!/bin/bash
#SBATCH -J special_permitl
#SBATCH --partition=amd
#SBATCH -t 8-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-2%1 # Creates 9 tasks and limits to 3 concurrent jobs

# your code goes below
module load any/python/3.8.3-conda
conda activate lstm-caise23

# Define an array of commands
declare -a commands=(


#"python ./dg_training.py -f BPI_Challenge_2013_closed_problems.csv -m lstm -e 50 -o rand_hpc" 
#"python ./dg_training.py -f BPI_2012_W_complete.csv -m lstm -e 50 -o rand_hpc" 

"python ./dg_training.py -f PermitLog_filtered_preprocessed.csv -m lstm -e 50 -o rand_hpc" 
#"python ./dg_training.py -f PrepaidTravelCost_preprocessed.csv -m lstm -e 50 -o rand_hpc" 
#"python ./dg_training.py -f RequestForPayment_preprocessed.csv -m lstm -e 50 -o rand_hpc" 




)

# Execute the command corresponding to this array job
${commands[$SLURM_ARRAY_TASK_ID]}




