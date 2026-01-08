#!/bin/bash
#SBATCH -J start
#SBATCH --partition=intel
#SBATCH -t 8-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --array=0-8%8 # Creates 9 tasks and limits to 3 concurrent jobs

# your code goes below
module load any/python/3.8.3-conda
conda activate lstm-caise23

# Define an array of commands
declare -a commands=(



"python ./dg_training.py -f BPI_Challenge_2012_W_Two_TS.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f BPI_Challenge_2017_W_Two_TS.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f confidential_1000.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f confidential_2000.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f ConsultaDataMining201618.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f cvs_pharmacy.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f Production.csv -m lstm_cx -e 5 -o bayesian"
"python ./dg_training.py -f PurchasingExample.csv -m lstm_cx -e 5 -o bayesian"

)

# Execute the command corresponding to this array job
${commands[$SLURM_ARRAY_TASK_ID]}




