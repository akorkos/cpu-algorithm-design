#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome00
#SBATCH --exclusive
#SBATCH -o ex02out.txt

# ./build/reduction-benchmark05v4 --benchmark_time_unit=us --benchmark_out_format=csv --benchmark_out=ex05.csv --benchmark_report_aggregates_only=true --benchmark_repetitions=50 \
#  --benchmark_min_time=0.001 --benchmark_min_warmup_time=0.01
export OMP_PLACES=numa_domains # threads for single-pass scan, otherwise numa_domains
export OMP_PROC_BIND=true

./benchmarkSIMDreduce
