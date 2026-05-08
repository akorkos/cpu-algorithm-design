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


add_executable( exec_SIMD_transform main_SIMD_transform.cpp )
rome_build( exec_SIMD_transform )

add_executable( exec_SIMD_reduce main_SIMD_reduce.cpp )
rome_build( exec_SIMD_reduce )


add_executable( benchmarkSIMDreduce SIMD_reduce_bench.cpp )
rome_build( benchmarkSIMDreduce )

add_executable( benchmarkSIMDtransform SIMD_transform_bench.cpp )
rome_build( benchmarkSIMDtransform )


add_executable( benchmarkSIMDtransformV SIMD_transformV_bench.cpp )
rome_build( benchmarkSIMDtransformV )

add_executable( benchmarkSIMDreduceV SIMD_reduceV_bench.cpp )
rome_build( benchmarkSIMDreduceV )