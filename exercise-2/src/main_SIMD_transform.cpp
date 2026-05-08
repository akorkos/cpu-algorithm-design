#include "SIMD_transform.hpp"
#include <iostream>

int main(void)
{
    SIMD_transform test;

  test.benchTransformIterator();
  test.benchTransformIteratorInnerLoop();
  test.benchTransformRange();
  test.benchTransformRangeInnerLoop();
  test.benchTransformStl();
  test.benchTransformSimdStl();
  test.benchXsimdTransform();
  test.benchXsimdTransformAligned();
  test.benchOmpSimdTransformIterator();
  test.benchOmpSimdTransformIteratorInnerLoop();
  test.benchOmpSimdTransformRange();
  test.benchOmpSimdTransformRangeInnerLoop();

  std::cout << test.get_log();
  return 0;
}
