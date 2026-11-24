#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_scan_fast.hpp"

int main(){
	using Scan = NUMA_omp::Scan<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Scan test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &Scan::scan_twopass, "scan_twopass", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::exclusive_scan_scalar, "exclusive_scan_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::inclusive_scan_scalar, "inclusive_scan_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::scan_scalar, "scan_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::scan_copy_if, "scan_copy_if", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::scan_partition1, "scan_partition1", Nl, Nh, dur, timings,rec);
	// benchmarker(&test, &Sc  an::scan_partition2, "scan_partition2", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Scan::scan_unique_copy, "scan_unique_copy", Nl, Nh, dur, timings,rec);

	


    // std::cout << test.get_log();

	return 0;
}
