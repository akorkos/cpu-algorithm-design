// #include "Stdfunctions.hpp"
#include <numeric>
#include <cmath>
#include <vector>
#include <ranges>
#include <string_view>
#include "utilities.hpp"
// ---  benchmarking --- 
#pragma once

template<typename F, typename C>
static void benchmarker(C test, F the_func , std::string_view func_name, size_t bN, size_t eN, std::vector<long int> dur, auto timings, auto& rec){
  
  auto warming = 1;
  
  for(auto& i = bN; i != eN ; i++){
	  auto N = 1<<i;
	  auto bytes = (*test.*the_func)(N);
	  auto bytes_ = std::get<0>(bytes) + std::get<1>(bytes);
	  
	  std::for_each(itSeq(0+warming), itSeq(timings), [bytes_, rec=std::begin(rec),dur=std::begin(dur)](auto i){
		dur[i] = std::chrono::nanoseconds(rec[i+1]-rec[i]).count();
		//std::cout <<"time[ns]: "<< dur[i] << ' ';
		//std::cout << " [GB/s]: " << bytes_ / dur[i] << '\n';
	  });
	  auto average = reduce(begin(dur),end(dur))/double(timings-warming);
	  auto stdev = transform_reduce(begin(dur)+warming, end(dur), 0.0, std::plus{}, [average](auto a){return (a-average)*(a-average);});
          stdev = std::sqrt(stdev / double(timings - warming));
	  auto maximum = reduce(begin(dur)+warming, end(dur), LONG_MAX, [](auto a, auto b){return std::min(a, b);});
	  auto cv = stdev/average;
	  // name, size, avg throughput, avg time, stdev time, cv
	  std::cout << func_name << "," << (1<<i) << "," << bytes_ / average << "," << average << "," <<stdev << "," << stdev / average << "\n";
	  
	  }  // end of for loop
}
