#include "SIMD_reduce.hpp"
#include <iostream>

int main(void)
{
    SIMD_reduce test;

  test.benchReduceIterator();
  test.benchReduceRange();
  test.benchReduceStl();
  test.benchReduceSimdXHorizontal();
  test.benchReduceSimdXVertical();
  test.benchReduceSimdXAlign();
  test.benchReduceSimdOmpIterator();
  test.benchReduceSimdOmpRange();
  test.benchReduceSimdStl();

  std::cout << test.get_log();
  return 0;
}
