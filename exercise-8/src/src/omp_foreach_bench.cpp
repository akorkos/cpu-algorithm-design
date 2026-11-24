#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_foreach.hpp"

int main()
{
	using foreach = NUMA_omp::foreach<ClockRecorder>;
	int Nl = 29;
	int Nh = 30;
	auto timings = 100;
	ClockRecorder rec(timings + 1);		// recordings
	std::vector<long int> dur(timings); // durations in nano
	foreach
		test(rec);

	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	// part1

	benchmarker(&test, &foreach::Map_AoS_scalar_scalar, "Map_AoS_scalar_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Map_AoS_complex_complex, "Map_AoS_complex_complex", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Map_AoS_aos_nn_flat, "Map_AoS_aos_nn_flat", Nl, Nh, dur, timings,rec);

	benchmarker(&test, &foreach::Map_AoS_aos_nm_flat_spantmp, "Map_AoS_aos_nm_flat_spantmp", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Map_AoS_aos_tuple_tuple, "Map_AoS_aos_tuple_tuple", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &foreach::Map_AoS_aos_tuple_pair, "Map_AoS_aos_tuple_pair", Nl, Nh, dur, timings,rec);

	benchmarker(&test, &foreach::Map_SoA_soa_zip_tuple_tuple, "Map_SoA_soa_zip_tuple_tuple", Nl, Nh, dur, timings, rec);
	benchmarker(&test, &foreach::Map_SoA_soa_zip_nm_spantmp, "Map_SoA_soa_zip_nm_spantmp", Nl, Nh, dur, timings, rec);
	benchmarker(&test, &foreach::Map_SoA_soa_explicit_nm_flat_spantmp, "Map_SoA_soa_explicit_nm_flat_spantmp", Nl, Nh, dur, timings, rec);


	std::cout << test.get_log();

	return 0;
}
