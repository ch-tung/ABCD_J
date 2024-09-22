#!/bin/bash
#SBATCH -A MAT226
#SBATCH -t 24:00:00
#SBATCH -N 2
#SBATCH -J Al_vacancy_migration

module purge
module load gcc/10.3.0  openmpi/4.1.2
export OMP_NUM_THREADS=1
# module load lammps

for temp in 810 870 910 950
do
    srun -N1 -n16 --exclusive lmp_mpi -var temp ${temp} -in in.Al_vacancy_migration \
    &
done

wait