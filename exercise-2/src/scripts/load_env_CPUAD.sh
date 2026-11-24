echo "Setting up environment variables"
spack load python@3.10.10%gcc@12.2.0
spack load gcc@14.2.0
spack load intel-oneapi-compilers@2024.2.1
spack load intel-oneapi-tbb@2021.12.0
spack load benchmark@git.v1.8.3=1.8.3
spack load cmake@3.27.7%gcc@13.1.0
spack load xsimd


export CC=icx
export CXX=icpx

# this update of the variable is necessary to run
export LD_LIBRARY_PATH=/opt/asc/spack/opt/spack/linux-debian12-x86_64_v2/gcc-12.2.0/gcc-14.2.0-duw5n2gdhts3cjt6ikatgoh6g5qnibj3/lib64:/opt/asc/spack/opt/spack/linux-debian12-x86_64_v2/gcc-12.2.0/gcc-14.2.0-duw5n2gdhts3cjt6ikatgoh6g5qnibj3/lib64:/shares/asc-opt/spack/opt/spack/linux-debian12-x86_64_v2/gcc-14.2.0/intel-oneapi-tbb-2021.12.0-uuzuputvf2xnv4obpp3ybhhhyqka4lj5/tbb/2021.12/lib/intel64/gcc4.8:/shares/asc-opt/spack/opt/spack/linux-debian12-x86_64_v2/gcc-14.2.0/intel-oneapi-compilers-2024.2.1-cs4fy4zo5nwzv5kqhki2qxjjw57pi32t/compiler/2024.2/opt/compiler/lib:/shares/asc-opt/spack/opt/spack/linux-debian12-x86_64_v2/gcc-14.2.0/intel-oneapi-compilers-2024.2.1-cs4fy4zo5nwzv5kqhki2qxjjw57pi32t/compiler/2024.2/lib:.:/opt/asc/pub/spack/opt/spack/linux-centos7-x86_64_v2/gcc-11.3.0/gcc-12.1.0-xz357upi6s7gnw23fmjob2oeqvawnp4q/lib64
# threads for single-pass scan, otherwise numa_domains
export OMP_PLACES=numa_domains 
export OMP_PROC_BIND=true

