#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "reduce_LoopUnrolling.hpp"

int main(){
	using reduce = reduce_LoopUnrolling<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    reduce_LoopUnrolling<ClockRecorder> test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &reduce::benchReduceOmpSimd, "benchReduceOmpSimd", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceDirectiveUnroll, "benchReduceDirectiveUnroll", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceDirectiveUnrollFactor64, "benchReduceDirectiveUnrollFactor64", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollManual, "benchReduceUnrollManual", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollLoopPeeling, "benchReduceUnrollLoopPeeling", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollLoopPeelingDirective, "benchReduceUnrollLoopPeelingDirective", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollTree, "benchReduceUnrollTree", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollTreeDirective, "benchReduceUnrollTreeDirective", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollSimdXHorizontal, "benchReduceUnrollSimdXHorizontal", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &reduce::benchReduceUnrollSimdXVertical, "benchReduceUnrollSimdXVertical", Nl, Nh, dur, timings,rec);
	

    // std::cout << test.get_log();

	return 0;
}
