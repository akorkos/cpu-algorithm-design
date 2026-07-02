#include "omp_foreach.hpp"
#include <iostream>


#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::foreach test;

 	test.Reordering_reverse();
	test.Reordering_gather_data();
	test.Reordering_scatter_func_n1();
	test.Subrange_singles();
	test.Subrange_interval();
	test.Subrange_intervals();
	test.Subrange_soa_explicit_stride_domain();



  std::cout << test.get_log();
  return 0;
}
