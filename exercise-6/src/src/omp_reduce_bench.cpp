#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_reduce.hpp"

int main(){
	using Reduce = NUMA_omp::Reduce<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Reduce test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &Reduce::reduce_scalar_small, "reduce_scalar_small", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_scalar, "reduce_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_complex_prod, "reduce_complex_prod", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_scalar_norm2, "reduce_scalar_norm2", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_n1, "reduce_n1", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_1m, "reduce_1m", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_nm, "reduce_nm", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_21, "reduce_21", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_pi_grid, "reduce_pi_grid", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Reduce::reduce_pi_rand, "reduce_pi_rand", Nl, Nh, dur, timings,rec);
	


    std::cout << test.get_log();

	return 0;
}
