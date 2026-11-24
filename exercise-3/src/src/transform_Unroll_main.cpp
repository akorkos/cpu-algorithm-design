#include "transform_LoopUnrolling.hpp"
#include <iostream>

int main(void)
{
  transform_LoopUnrolling test;

  test.benchTransformOmpSimd(256);
  test.benchTransformDirectiveUnroll(256);
  test.benchTransformDirectiveUnrollFactor64(256);
  test.benchTransformUnrollManual(256);
  test.benchTransformUnrollLoopPeeling(256);
  test.benchTransformUnrollLoopPeelingDirective(256);


  std::cout << test.get_log();
  return 0;
}
