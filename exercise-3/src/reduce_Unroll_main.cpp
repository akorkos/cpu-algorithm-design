#include "reduce_LoopUnrolling.hpp"
#include <iostream>

int main(void)
{
  reduce_LoopUnrolling test;

  test.benchReduceOmpSimd(256);
  test.benchReduceDirectiveUnroll(256);
  test.benchReduceDirectiveUnrollFactor64(256);
  test.benchReduceUnrollManual(256);
  test.benchReduceUnrollLoopPeeling(256);
  test.benchReduceUnrollLoopPeelingDirective(256);
  test.benchReduceUnrollTree(256);
  test.benchReduceUnrollTreeDirective(256);
  test.benchReduceUnrollSimdXHorizontal(256);
  test.benchReduceUnrollSimdXVertical(256);


  std::cout << test.get_log();
  return 0;
}
