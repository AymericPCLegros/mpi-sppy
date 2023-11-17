#!/bin/bash

SOLVERNAME=gurobi

#python agnostic_cylinders.py --help

#mpiexec -np 3 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --xhatshuffle --lagrangian --rel-gap 0.01

python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=3 

#mpiexec -np 2 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=10 --lagrangian --rel-gap 0.01

#mpiexec -np 2 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=10 --xhatshuffle --rel-gap 0.01