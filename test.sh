sbatch --ntasks=1  --cpus-per-task=1  --output=./out/out-01.log  --job-name=eph-fcigf run.sh
sbatch --ntasks=1  --cpus-per-task=4  --output=./out/out-04.log  --job-name=eph-fcigf run.sh
sbatch --ntasks=1  --cpus-per-task=16 --output=./out/out-16.log  --job-name=eph-fcigf run.sh
sbatch --ntasks=1  --cpus-per-task=28 --output=./out/out-28.log  --job-name=eph-fcigf run.sh
