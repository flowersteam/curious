#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p court
#SBATCH -t 3:00:00
#SBATCH -e ./run_HER_trials0_0.sh.err
#SBATCH -o ./run_HER_trials0_0.sh.out
rm log.txt; 
export EXP_INTERP='/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3' ;
echo '=================> Her : Trial 0, 2018-08-06 13:04:22.429517';
echo '=================> Her : Trial 0, 2018-08-06 13:04:22.429517' >> log.txt;
$EXP_INTERP train.py  --replay_strategy future --log_dir ./save/FetchPush-v1/0/ & 
wait