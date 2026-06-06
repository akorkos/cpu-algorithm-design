#include "omp_reduce.hpp"
#include <iostream>


#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::Reduce test;

  // test.reduce_scalar_small();
  // test.reduce_scalar();
  // test.reduce_complex_prod();
  // test.reduce_scalar_norm2();
  // test.reduce_n1();
  // test.reduce_1m();
  // test.reduce_nm();
  // test.reduce_21();
  // test.reduce_boundingbox();
  test.reduce_pi_grid(1024*1024);

  std::cout << test.get_log();
  return 0;
}
