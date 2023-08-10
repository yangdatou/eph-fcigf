sbatch --ntasks=1   --cpus-per-task=1  --output=./out/out-01.log  --job-name=out-01 run.sh
sbatch --ntasks=4   --cpus-per-task=1  --output=./out/out-04.log  --job-name=out-04 run.sh
sbatch --ntasks=16  --cpus-per-task=1  --output=./out/out-16.log  --job-name=out-16 run.sh
