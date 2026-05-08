#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "SIMD_reduce.hpp"

int main(){
	using reduce = SIMD_reduce<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    SIMD_reduce<ClockRecorder> test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &reduce::benchReduceIterator, "benchReduceIterator", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceRange, "benchReduceRange", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceStl, "benchReduceStl", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdStl, "benchReduceSimdStl", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdOmpIterator, "benchReduceSimdOmpIterator", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdOmpRange, "benchReduceSimdOmpRange", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdXHorizontal, "benchReduceSimdXHorizontal", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdXVertical, "benchReduceSimdXVertical", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceSimdXAlign, "benchReduceSimdXAlign", Nl, Nh, dur, timings,rec);

    // std::cout << test.get_log();

	return 0;
}
