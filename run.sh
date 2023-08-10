#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem=0
#SBATCH --output=out.log

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2

export NCORES=$SLURM_CPUS_PER_TASK;
export OMP_NUM_THREADS=$NCORES;
export MKL_NUM_THREADS=$NCORES
export OPENBLAS_NUM_THREADS=$NCORES
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;

echo OMP_NUM_THREADS      = $OMP_NUM_THREADS
echo MKL_NUM_THREADS      = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY     = $PYSCF_MAX_MEMORY

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

mkdir -p /scratch/global/yangjunjie/$SLURM_JOB_NAME-$SLURM_JOB_ID
export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME-$SLURM_JOB_ID
export PYSCF_TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME-$SLURM_JOB_ID

export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/
export PYTHONPATH=/home/yangjunjie/work/cc-eph/cceph-main/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/epcc-hol/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/wick-dev/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/cqcpy-master/:$PYTHONPATH

export PYTHONUNBUFFERED=TRUE;
time python eph-fcigf.py

