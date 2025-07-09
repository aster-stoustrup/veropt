#!/bin/bash -l
#SBATCH -p {partition_name}
#SBATCH -A {group_name}
#SBATCH --job-name={simulation_id}
#SBATCH --time=23:59:59
#SBATCH --constraint={constraint}
#SBATCH --nodes=1
#SBATCH --ntasks={n_cores}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
##SBATCH --threads-per-core=1
##SBATCH --exclusive
#SBATCH --output={slurm_log_filename}.out

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
    echo "SLURM_JOB_ID = $SLURM_JOB_ID"
fi

ml purge

source /software/ocean/software_gcc2024/py3.11/miniconda3/etc/profile.d/conda.sh

conda activate veros_jax_cpu

veros resubmit -i {output_filename} -n {n_cycles} -l {cycle_length} \
-c 'mpiexec -n {n_cores} -- veros run {run_script_filename}.py -b {backend} -n {n_cores_nx} {n_cores_ny} --float-type {float_type}' \
--callback 'sbatch {batch_script_filename}.sh'