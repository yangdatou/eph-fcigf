sbatch --ntasks=1  --output=out-1.log  --job-name=eph-fcigf run.sh
sbatch --ntasks=4  --output=out-4.log  --job-name=eph-fcigf run.sh
sbatch --ntasks=16 --output=out-16.log --job-name=eph-fcigf run.sh
sbatch --ntasks=28 --output=out-28.log --job-name=eph-fcigf run.sh
