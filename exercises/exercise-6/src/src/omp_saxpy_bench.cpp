#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_saxpy.hpp"

int main(){
	using Saxpy = NUMA_omp::Saxpy<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Saxpy test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &Saxpy::transform_ops_value, "transform_ops_value", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_functor_value, "transform_functor_value", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_lambda_value, "transform_lambda_value", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_lambda_index, "transform_lambda_index", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_lambda_bad_init, "transform_lambda_bad_init", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_lambda_no_abstraction, "transform_lambda_no_abstraction", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_mandelbrot_fixed, "transform_mandelbrot_fixed", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Saxpy::transform_mandelbrot, "transform_mandelbrot", Nl, Nh, dur, timings,rec);
	


    std::cout << test.get_log();

	return 0;
}
