#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "SIMD_transform.hpp"

int main(){
	using transform = SIMD_transform<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    SIMD_transform<ClockRecorder> test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &transform::benchTransformIterator, "benchTransformIterator", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformIteratorInnerLoop, "benchTransformIteratorInnerLoop", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformRange, "benchTransformRange", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformRangeInnerLoop, "benchTransformRangeInnerLoop", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformStl, "benchTransformStl", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformSimdStl, "benchTransformSimdStl", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchXsimdTransform, "benchXsimdTransform", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchXsimdTransformAligned, "benchXsimdTransformAligned", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchOmpSimdTransformIterator, "benchOmpSimdTransformIterator", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchOmpSimdTransformIteratorInnerLoop, "benchOmpSimdTransformIteratorInnerLoop", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchOmpSimdTransformRange, "benchOmpSimdTransformRange", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchOmpSimdTransformRangeInnerLoop, "benchOmpSimdTransformRangeInnerLoop", Nl, Nh, dur, timings,rec);
    // std::cout << test.get_log();

	return 0;
}
