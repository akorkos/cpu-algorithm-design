#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_foreach.hpp"

int main()
{
	using foreach = NUMA_omp::foreach<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
	auto timings = 100;
	ClockRecorder rec(timings + 1);		// recordings
	std::vector<long int> dur(timings); // durations in nano
	foreach
		test(rec);

	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";   
	benchmarker(&test, &foreach::Reordering_reverse, "Reordering_reverse", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Reordering_gather_data, "Reordering_gather_data", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Reordering_scatter_func_n1, "Reordering_scatter_func_n1", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Subrange_singles, "Subrange_singles", Nl, Nh, dur, timings, rec);
	benchmarker(&test, &foreach::Subrange_interval, "Subrange_interval", Nl, Nh, dur, timings, rec);
	benchmarker(&test, &foreach::Subrange_intervals, "Subrange_intervals", Nl, Nh, dur, timings, rec);
	benchmarker(&test, &foreach::Subrange_soa_explicit_stride_domain, "Subrange_soa_explicit_stride_domain", Nl, Nh, dur, timings, rec);


	std::cout << test.get_log();

	return 0;
}