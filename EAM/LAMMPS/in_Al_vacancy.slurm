#!/bin/bash
#SBATCH --job-name=lammps_job
#SBATCH --output=lammps_output_%j.txt
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=standard

module load lammps

for temp in 810 870 910 950
do
    jsrun -n 1 -a 1 -c 4 -g 1 lmp_mpi -var temp ${temp} -i in.Al_vacancy_migration &
done

wait