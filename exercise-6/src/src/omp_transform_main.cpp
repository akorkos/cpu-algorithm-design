#include "omp_saxpy.hpp"
#include <iostream>


#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::Saxpy test;

  test.transform_ops_value(1<<30);

  std::cout << test.get_log();
  return 0;
}
