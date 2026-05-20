cmake_minimum_required( VERSION 3.14 )
project(NUMA CXX)

# find_package( benchmark REQUIRED )
find_package( xsimd REQUIRED)
find_package( OpenMP REQUIRED COMPONENTS CXX)

function( rome_build targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_23 )
	target_compile_options( ${targetname} PRIVATE -march=core-avx2 -mtune=core-avx2 -qopenmp -qopenmp-simd -O3 -mfma ) 
	target_include_directories( ${targetname} PRIVATE include . )
	target_link_libraries( ${targetname} PRIVATE OpenMP::OpenMP_CXX xsimd)
endfunction()


add_executable( exec_transform_Unroll transform_Unroll_main.cpp )
rome_build( exec_transform_Unroll )

add_executable( exec_reduce_Unroll reduce_Unroll_main.cpp )
rome_build( exec_reduce_Unroll )


add_executable( reducebenchmarkUnroll reduce_Unroll_bench.cpp )
rome_build( reducebenchmarkUnroll )

add_executable( reduceVbenchmarkUnroll reduceV_Unroll_bench.cpp )
rome_build( reduceVbenchmarkUnroll )


add_executable( transformbenchmarkUnroll transform_Unroll_bench.cpp )
rome_build( transformbenchmarkUnroll )

add_executable( transformVbenchmarkUnroll transformV_Unroll_bench.cpp )
rome_build( transformVbenchmarkUnroll )

add_executable( benchUnrollFactor benchUnrollFactor.cpp )
rome_build( benchUnrollFactor)