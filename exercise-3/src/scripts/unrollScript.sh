#!/bin/bash


unrollfactor[0]="#define UNROLLFACTOR 8"
unrollfactor[1]="#define UNROLLFACTOR 16"
unrollfactor[2]="#define UNROLLFACTOR 32"
unrollfactor[3]="#define UNROLLFACTOR 64"
unrollfactor[4]="#define UNROLLFACTOR 96"
unrollfactor[5]="#define UNROLLFACTOR 128"
unrollfactor[6]="#define UNROLLFACTOR 160"
unrollfactor[7]="#define UNROLLFACTOR 192"
unrollfactor[8]="#define UNROLLFACTOR 224"
unrollfactor[9]="#define UNROLLFACTOR 256"
unrollfactor_num[0]=8
unrollfactor_num[1]=16
unrollfactor_num[2]=32
unrollfactor_num[3]=64
unrollfactor_num[4]=96
unrollfactor_num[5]=128
unrollfactor_num[6]=160
unrollfactor_num[7]=192
unrollfactor_num[8]=224
unrollfactor_num[9]=256

length=${#unrollfactor[@]}
factorfile="../UNROLLFACTOR.h"
# set up directories


BIN_DIR="changingUnrollFactor"
mkdir -p "$BIN_DIR"
mkdir -p res
for ((i=0; i<length; i++))
do
    echo "building unroll factor ${unrollfactor_num[i]}"
    echo "#pragma once" > "${factorfile}"
    echo "${unrollfactor[i]}" >> "${factorfile}"
    make benchUnrollFactor
    mv benchUnrollFactor "$BIN_DIR/program_${unrollfactor_num[i]}"
    echo "#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome00
#SBATCH --exclusive
#SBATCH -o res/program_${unrollfactor_num[i]}.txt

export OMP_PLACES=numa_domains # threads for single-pass scan, otherwise numa_domains
export OMP_PROC_BIND=true

./$BIN_DIR/program_${unrollfactor_num[i]}" > submit_unrollfactor.sh
    sbatch submit_unrollfactor.sh 
    echo "${unrollfactor_num[i]} job submitted."
done

