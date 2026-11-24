// #include "omp_scan.hpp"
#include "omp_stencil.hpp"
#include <iostream>


#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::Stencil test;

  test.Stencil_1d_dirichlet_index(1<<22);
  test.Stencil_1d_neumann_index(1<<22);
  test.Stencil_1d_dirichlet_domain(1<<22);
  test.Stencil_2d_dirichlet_index(32);
  test.Stencil_2d_dirichlet_index_explicit(32);
  // test.Stencil_3d_dirichlet_index(256);
  test.Stencil_3d_dirichlet_index_explicit(256);
  // test.Stencil_3d_dirichlet_index_explicit_cacheBlocking(256);

  std::cout << test.get_log();
  return 0;
}