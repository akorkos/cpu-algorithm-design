// #include "omp_scan.hpp"
#include "omp_scan_fast.hpp"
#include <iostream>


#define implnamespace NUMA_omp;
int main(void)
{
  NUMA_omp::Scan test;

  test.scan_twopass(1<<22);
  test.exclusive_scan_scalar(1<<22);
  test.inclusive_scan_scalar(1<<22);
  test.scan_scalar(1<<22);
  test.scan_copy_if(1<<22);
  test.scan_partition1(1<<22);
  // test.scan_partition2();
  test.scan_unique_copy(1<<22);

  std::cout << test.get_log();
  return 0;
}
// Input:          ( 1, 1, 0, 1, 1, 0, 1, 1, 0, 1)
// exclusive_scan: ( 0, 1, 2, 2, 3, 4, 4, 5, 6, 6)
// inclusive_scan: ( 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)
// scan:           ( 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)
// Input:          ( 1,11,21, 1,11,21, 1,11,21, 1)
// scan copy_if I: ( 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)
// scan copy_if W: ( 1,11, 1,11, 1,11, 1)
// scan partition I:       ( 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)
// scan partition W:       7:( 1,11, 1,11, 1,11, 1,21,21,21)
// scan partition I0:      ( 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)
// scan partition I1:      ( 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3)
// scan partition W:       7:( 1,11, 1,11, 1,11, 1,21,21,21)
// scan unique I:  ( 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7)
// scan unique W:  ( 1,21, 1,21, 1,21, 1)