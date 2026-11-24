cmake_minimum_required( VERSION 3.14 )
project(NUMA CXX)

# find_package( benchmark REQUIRED )
find_package( xsimd REQUIRED)
find_package( OpenMP REQUIRED COMPONENTS CXX)
find_package( TBB REQUIRED )

function( rome_build targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_23 )
	target_compile_options( ${targetname} PRIVATE -march=znver2 -mtune=znver2 -fopenmp -fopenmp-simd -O3 -g -mfma) 
	# target_compile_options( ${targetname} PRIVATE -march=znver2 -mtune=znver2 -fopenmp -fopenmp-simd -O3 -g -mfma) 
	# target_compile_options( ${targetname} PRIVATE -march=znver2 -mtune=znver2 -fopenmp -fopenmp-simd  -g -mfma) 
	target_include_directories( ${targetname} PRIVATE include . )
	target_link_libraries( ${targetname} PRIVATE OpenMP::OpenMP_CXX xsimd TBB::tbb)
endfunction()




add_executable( benchstencil src/omp_stencil_bench.cpp )
rome_build( benchstencil )

add_executable( exec_stencil src/omp_stencil_main.cpp )
rome_build( exec_stencil )