#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p court
#SBATCH -t 3:00:00
#SBATCH -e ./run_HER_trials0_0.sh.err
#SBATCH -o ./run_HER_trials0_0.sh.out
rm log.txt; 
export EXP_INTERP='/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3' ;
echo '=================> Her : Trial 0, 2018-08-06 13:07:08.515231';
echo '=================> Her : Trial 0, 2018-08-06 13:07:08.515231' >> log.txt;
$EXP_INTERP train.py --logdir ./save/tests/0/ --env ModularArm012-v0 &
wait