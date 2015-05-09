#!/bin/sh
#PBS -l walltime=1:00:00
#PBS -l nodes=8:ppn=8
#PBS -l pmem=1gb
#PBS -N openmpi
cd $PBS_O_WORKDIR
. /opt/torque/etc/openmpi-setup.sh
export OMP_NUM_THREADS=6

#mpirun -mca btl tcp,self -mca plm_rsh_agent ssh --map-by node ./pals 1 20 ./train-data 943 1682 50
#mpirun -mca btl tcp,self -mca plm_rsh_agent ssh --map-by ppr:1:node ./pals 1 20 ./data/train-data 71567 65353 50
qstat -f $PBS_JOBID | grep -i resource

#mpirun -np 4 ./pals 1 20 ./data/train-data 71567 65353 50
#mpirun -mca btl tcp,self -mca plm_rsh_agent ssh --map-by node -np 4 ./pals 1 20 ./data/train-data 71567 65353 50
