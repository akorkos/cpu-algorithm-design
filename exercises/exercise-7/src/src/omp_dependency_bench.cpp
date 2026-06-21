#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "sol_omp_dependency.hpp"

int main(){
	using Dependency = NUMA_omp::Dependency<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 10;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Dependency test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &Dependency::update_sequential, "update_sequential", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::update_parallel_simd, "update_parallel_simd", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::update_parallel_nested_loop, "update_parallel_nested_loop", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::between_iter_sequential, "between_iter_sequential", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::between_iter_twoloops, "between_iter_twoloops", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::between_iter_optimal, "between_iter_optimal", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::between_iter_manual_split, "between_iter_manual_split", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Dependency::between_iter_task, "between_iter_task", Nl, Nh, dur, timings,rec);


    // std::cout << test.get_log();

	return 0;
}
