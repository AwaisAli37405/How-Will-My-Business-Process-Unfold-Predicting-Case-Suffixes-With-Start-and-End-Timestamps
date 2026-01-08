#!/bin/bash
#SBATCH -J specialpermit
#SBATCH --partition=amd
#SBATCH -t 8-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --array=0-10%6 # Creates 9 tasks and limits to 3 concurrent jobs

# your code goes below
module load any/python/3.8.3-conda
conda activate lstm-caise23

# Define an array of commands
declare -a commands=(




"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PermitLog_filtered_preprocessed -b "PermitLog_filtered_preprocessed.h5" -v "arg_max" -r 1"
"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PermitLog_filtered_preprocessed -b "PermitLog_filtered_preprocessed.h5" -v "topk" -r 1"
"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PermitLog_filtered_preprocessed -b "PermitLog_filtered_preprocessed.h5" -v "d_action" -r 1" 
"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PermitLog_filtered_preprocessed -b "PermitLog_filtered_preprocessed.h5" -v "random_choice" -r 1"
"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PermitLog_filtered_preprocessed -b "PermitLog_filtered_preprocessed.h5" -v "nucleus" -r 1"



#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PrepaidTravelCost_preprocessed -b "PrepaidTravelCost_preprocessed.h5" -v "arg_max" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PrepaidTravelCost_preprocessed -b "PrepaidTravelCost_preprocessed.h5" -v "topk" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PrepaidTravelCost_preprocessed -b "PrepaidTravelCost_preprocessed.h5" -v "d_action" -r 1" 
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PrepaidTravelCost_preprocessed -b "PrepaidTravelCost_preprocessed.h5" -v "random_choice" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c PrepaidTravelCost_preprocessed -b "PrepaidTravelCost_preprocessed.h5" -v "nucleus" -r 1"




#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c RequestForPayment_preprocessed -b "RequestForPayment_preprocessed.h5" -v "arg_max" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c RequestForPayment_preprocessed -b "RequestForPayment_preprocessed.h5" -v "topk" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c RequestForPayment_preprocessed -b "RequestForPayment_preprocessed.h5" -v "d_action" -r 1" 
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c RequestForPayment_preprocessed -b "RequestForPayment_preprocessed.h5" -v "random_choice" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c RequestForPayment_preprocessed -b "RequestForPayment_preprocessed.h5" -v "nucleus" -r 1"






#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_Challenge_2013_closed_problems -b "BPI_Challenge_2013_closed_problems.h5" -v "arg_max" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_Challenge_2013_closed_problems -b "BPI_Challenge_2013_closed_problems.h5" -v "topk" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_Challenge_2013_closed_problems -b "BPI_Challenge_2013_closed_problems.h5" -v "d_action" -r 1" 
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_Challenge_2013_closed_problems -b "BPI_Challenge_2013_closed_problems.h5" -v "random_choice" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_Challenge_2013_closed_problems -b "BPI_Challenge_2013_closed_problems.h5" -v "nucleus" -r 1"


#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_2012_W_complete -b "BPI_2012_W_complete.h5" -v "arg_max" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_2012_W_complete -b "BPI_2012_W_complete.h5" -v "topk" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_2012_W_complete -b "BPI_2012_W_complete.h5" -v "d_action" -r 1" 
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_2012_W_complete -b "BPI_2012_W_complete.h5" -v "random_choice" -r 1"
#"python ./dg_predictiction-onetimestamp.py -a pred_sfx -c BPI_2012_W_complete -b "BPI_2012_W_complete.h5" -v "nucleus" -r 1"





)

# Execute the command corresponding to this array job
${commands[$SLURM_ARRAY_TASK_ID]}




 