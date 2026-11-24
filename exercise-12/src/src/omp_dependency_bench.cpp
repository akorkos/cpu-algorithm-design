#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "sol_omp_pipeline.hpp"

int main(){
	using Pipeline = NUMA_omp::Pipeline<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 10;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Pipeline test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	// benchmarker(&test, &Pipeline::update_sequential, "update_sequential", Nl, Nh, dur, timings,rec);
	// benchmarker(&test, &Pipeline::update_parallel_simd, "update_parallel_simd", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Pipeline::update_parallel_nested_loop, "update_parallel_nested_loop", Nl, Nh, dur, timings,rec);


    // std::cout << test.get_log();

	return 0;
}
