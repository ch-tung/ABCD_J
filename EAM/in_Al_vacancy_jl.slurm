#!/bin/bash
#SBATCH -A MAT226
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J Al_vacancy_migration

module purge

srun -A MAT226 -N1 -n4 julia ./Al_vacancy_migration.jl

wait