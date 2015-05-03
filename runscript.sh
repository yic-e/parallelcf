#PBS -lwalltime=1:00:00      # One hour runtime
#PBS -lnodes=2:ppn=2         # 2 nodes with 2 cpus each
#PBS -lpmem=1gb              # 1 GB memory per cpu
#PBS -Nmpi-verify-openmpi    # the name of the job
#PBS -o ./log
#PBS -e ./log
cd /home/yicheng1/pp-final/parallelcf

/opt/mpiexec/bin/mpiexec -np 2 ./pals 5 ./train-data 943 1682 50

qstat -f $PBS_JOBID | grep -i resource  
