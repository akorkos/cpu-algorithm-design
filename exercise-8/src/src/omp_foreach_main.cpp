#include "omp_foreach.hpp"
#include <iostream>

#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::foreach test;
  test.Map_AoS_scalar_scalar();
  test.Map_AoS_complex_complex();
  test.Map_AoS_aos_nn_flat();

  test.Map_AoS_aos_nm_flat_spantmp();
  test.Map_AoS_aos_tuple_tuple();
  test.Map_AoS_aos_tuple_pair();

  test.Map_SoA_soa_zip_tuple_tuple();
  test.Map_SoA_soa_zip_nm_spantmp();
  test.Map_SoA_soa_explicit_nm_flat_spantmp();
  std::cout << test.get_log();
  return 0;
}
