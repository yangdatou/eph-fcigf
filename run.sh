#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem=0
#SBATCH --job-name=eph-fcigf
#SBATCH --exclude=pauling013
#SBATCH --output=/scratch/global/yangjunjie/slurm-%x-%j.log

export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME-$SLURM_JOB_ID/
export PYSCF_TMPDIR=TMPDIR
echo $TMPDIR
mkdir -p $TMPDIR

export LOG_TMPDIR=$SLURM_SUBMIT_DIR/out/$SLURM_JOB_NAME-$SLURM_JOB_ID/
echo $LOG_TMPD
mkdir -p $LOG_TMPD

echo TMPDIR=$TMPDIR
echo LOG_TMPDIR=$LOG_TMPDIR

ln -s /scratch/global/yangjunjie/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.log $LOG_TMPDIR/slurm.out

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE

echo SLURM_NTASKS         = $SLURM_NTASKS
echo OMP_NUM_THREADS      = $OMP_NUM_THREADS
echo MKL_NUM_THREADS      = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY     = $PYSCF_MAX_MEMORY

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/
export PYTHONPATH=/home/yangjunjie/work/cc-eph/cceph-main/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/epcc-hol/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/wick-dev/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/cqcpy-master/:$PYTHONPATH

for nmpi in 64 16 04 01; do
    echo nmpi=$nmpi
    export LOG_TMPDIR=$SLURM_SUBMIT_DIR/out/$SLURM_JOB_NAME-$SLURM_JOB_ID/$nmpi/
    mkdir -p $LOG_TMPDIR

    time mpirun -n $nmpi python main.py
done

cp /scratch/global/yangjunjie/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.log $LOG_TMPDIR/slurm.out

