#!/bin/sh
#PBS -l walltime=1:00:00
#PBS -l nodes=8:ppn=2
#PBS -l pmem=1gb
#PBS -N intelmpi
cd $PBS_O_WORKDIR
#. /opt/torque/etc/openmpi-setup.sh
#mpirun -mca btl tcp,self -mca plm_rsh_agent ssh --map-by node ./pals 1 20 ./train-data 943 1682 50
#/opt/intel/impi/5.0.3.048/intel64/bin/mpirun ./pals 1 32 ./data/final-train 69878 10677 20
/opt/intel/impi/5.0.3.048/intel64/bin/mpirun ./pals 1 128 ml2m/ml-20m 138493 26744 50 -1
qstat -f $PBS_JOBID | grep -i resource
#mpirun -np 4 ./pals 1 20 ./data/train-data 71567 65353 50
#mpirun -mca btl tcp,self -mca plm_rsh_agent ssh --map-by node -np 4 ./pals 1 20 ./data/train-data 71567 65353 50
