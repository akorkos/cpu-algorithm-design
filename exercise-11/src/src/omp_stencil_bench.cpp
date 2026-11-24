#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "omp_stencil.hpp"

int main(){
	using Stencil = NUMA_omp::Stencil<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 20;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    Stencil test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	benchmarker(&test, &Stencil::Stencil_1d_dirichlet_index, "Stencil_1d_dirichlet_index", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Map_AoS_scalar_scalar, "Map_AoS_scalar_scalar", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Stencil_1d_neumann_index, "Stencil_1d_neumann_index", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Stencil_1d_dirichlet_domain, "Stencil_1d_dirichlet_domain", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Stencil_2d_dirichlet_index, "Stencil_2d_dirichlet_index", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Stencil_2d_dirichlet_index_explicit, "Stencil_2d_dirichlet_index_explicit", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &Stencil::Stencil_3d_dirichlet_index_explicit, "Stencil_3d_dirichlet_index_explicit", Nl, Nh, dur, timings,rec);
	// benchmarker(&test, &Stencil::Stencil_3d_dirichlet_index_explicit_cacheBlocking, "Stencil_3d_dirichlet_index_explicit", Nl, Nh, dur, timings,rec);



    std::cout << test.get_log();

	return 0;
}
