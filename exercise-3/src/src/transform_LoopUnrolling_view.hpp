#include "UNROLLFACTOR.h"
#include "utilities.hpp"

#include <algorithm>
#include <cassert>
#include <complex>
#include <deque>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <ranges>
#include <utility>
#include <vector>
#include <xsimd/xsimd.hpp>

#pragma once

using namespace std;

template<typename R = array<char, 1>, typename L = function<void(void)>>
class transform_LoopUnrolling_view {
  private:
    static array<char, 1> p_one;
    stringstream          p_log;
    R                    &p_loop_state;
    L                     p_loop_action;

  public:
    transform_LoopUnrolling_view(
      R &loop_state = p_one, L loop_action = []() {})
        : p_loop_state(loop_state), p_loop_action(loop_action) {
      p_log << setprecision(3);
    }

    string get_log() { return p_log.str(); }

    using Index                          = int;
    using Int                            = int32_t;
    using Real                           = float;
    using CReal                          = complex<Real>;
    template<typename T> using Container = vector<T, allocator<T>>;
    template<typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec =
      execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec =
      execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n    = 3;
    static constexpr Index default_m    = 2;
    static constexpr Index default_N    = 24;
    static constexpr Index default_Nout = 20;

    auto benchTransformOmpSimd(Index N = default_N) {
      // Do not change
      Real           a             = -1.0f;
      constexpr auto unroll_factor = 64;
      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });
      std::vector<Real> W(256, 10);

      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
        // using openmp simd directive with index
#pragma omp simd
        for (Index i = 0; i < N; i++) { W[i % 256] = a * V[i] + W[i % 256]; }
        p_loop_action();
      }
      p_log << "OmpIndex \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnroll(Index N = default_N) {
      // Do not change
      Real           a             = -1.0f;
      constexpr auto unroll_factor = 64;
      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });
      std::vector<Real> W(256, 10);

      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
        // using openmp unroll directive with index
#pragma omp simd
#pragma unroll
        for (Index i = 0; i < N; i++) { W[i % 256] = a * V[i] + W[i % 256]; }
        p_loop_action();
      }
      p_log << "DirectiveUnroll \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnrollFactor64(Index N = default_N) {
      // Do not change
      Real           a             = -1.0f;
      constexpr auto unroll_factor = 64;
      N                            = N % unroll_factor ? N : N + 1;
      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });
      std::vector<Real> W(256, 10);

      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
        // using openmp unroll directive with index
#pragma omp simd
#pragma unroll(unroll_factor)
        for (Index i = 0; i < N; i++) { W[i % 256] = a * V[i] + W[i % 256]; }
        p_loop_action();
      }
      p_log << "DirectiveUnrollFactor64 \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollManual(Index N = default_N) {
      // Do not change
      Real           a             = -1.0f;
      constexpr auto unroll_factor = 64;
      N                            = N % unroll_factor ? N : N + 1;
      auto rem                     = N % unroll_factor;
      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });
      std::vector<Real> W(256, 10);

      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
        // Do not change
        // using manual unroll with index
#pragma omp simd
        for (Index i = 0; i < N; i += unroll_factor) {
          if (i + 0 < N) { W[(i + 0) % 256] = a * V[i + 0] + W[(i + 0) % 256]; }
          if (i + 1 < N) { W[(i + 1) % 256] = a * V[i + 1] + W[(i + 1) % 256]; }
          if (i + 2 < N) { W[(i + 2) % 256] = a * V[i + 2] + W[(i + 2) % 256]; }
          if (i + 3 < N) { W[(i + 3) % 256] = a * V[i + 3] + W[(i + 3) % 256]; }
          if (i + 4 < N) { W[(i + 4) % 256] = a * V[i + 4] + W[(i + 4) % 256]; }
          if (i + 5 < N) { W[(i + 5) % 256] = a * V[i + 5] + W[(i + 5) % 256]; }
          if (i + 6 < N) { W[(i + 6) % 256] = a * V[i + 6] + W[(i + 6) % 256]; }
          if (i + 7 < N) { W[(i + 7) % 256] = a * V[i + 7] + W[(i + 7) % 256]; }
          if (i + 8 < N) { W[(i + 8) % 256] = a * V[i + 8] + W[(i + 8) % 256]; }
          if (i + 9 < N) { W[(i + 9) % 256] = a * V[i + 9] + W[(i + 9) % 256]; }
          if (i + 10 < N) {
            W[(i + 10) % 256] = a * V[i + 10] + W[(i + 10) % 256];
          }
          if (i + 11 < N) {
            W[(i + 11) % 256] = a * V[i + 11] + W[(i + 11) % 256];
          }
          if (i + 12 < N) {
            W[(i + 12) % 256] = a * V[i + 12] + W[(i + 12) % 256];
          }
          if (i + 13 < N) {
            W[(i + 13) % 256] = a * V[i + 13] + W[(i + 13) % 256];
          }
          if (i + 14 < N) {
            W[(i + 14) % 256] = a * V[i + 14] + W[(i + 14) % 256];
          }
          if (i + 15 < N) {
            W[(i + 15) % 256] = a * V[i + 15] + W[(i + 15) % 256];
          }
          if (i + 16 < N) {
            W[(i + 16) % 256] = a * V[i + 16] + W[(i + 16) % 256];
          }
          if (i + 17 < N) {
            W[(i + 17) % 256] = a * V[i + 17] + W[(i + 17) % 256];
          }
          if (i + 18 < N) {
            W[(i + 18) % 256] = a * V[i + 18] + W[(i + 18) % 256];
          }
          if (i + 19 < N) {
            W[(i + 19) % 256] = a * V[i + 19] + W[(i + 19) % 256];
          }
          if (i + 20 < N) {
            W[(i + 20) % 256] = a * V[i + 20] + W[(i + 20) % 256];
          }
          if (i + 21 < N) {
            W[(i + 21) % 256] = a * V[i + 21] + W[(i + 21) % 256];
          }
          if (i + 22 < N) {
            W[(i + 22) % 256] = a * V[i + 22] + W[(i + 22) % 256];
          }
          if (i + 23 < N) {
            W[(i + 23) % 256] = a * V[i + 23] + W[(i + 23) % 256];
          }
          if (i + 24 < N) {
            W[(i + 24) % 256] = a * V[i + 24] + W[(i + 24) % 256];
          }
          if (i + 25 < N) {
            W[(i + 25) % 256] = a * V[i + 25] + W[(i + 25) % 256];
          }
          if (i + 26 < N) {
            W[(i + 26) % 256] = a * V[i + 26] + W[(i + 26) % 256];
          }
          if (i + 27 < N) {
            W[(i + 27) % 256] = a * V[i + 27] + W[(i + 27) % 256];
          }
          if (i + 28 < N) {
            W[(i + 28) % 256] = a * V[i + 28] + W[(i + 28) % 256];
          }
          if (i + 29 < N) {
            W[(i + 29) % 256] = a * V[i + 29] + W[(i + 29) % 256];
          }
          if (i + 30 < N) {
            W[(i + 30) % 256] = a * V[i + 30] + W[(i + 30) % 256];
          }
          if (i + 31 < N) {
            W[(i + 31) % 256] = a * V[i + 31] + W[(i + 31) % 256];
          }
          if (i + 32 < N) {
            W[(i + 32) % 256] = a * V[i + 32] + W[(i + 32) % 256];
          }
          if (i + 33 < N) {
            W[(i + 33) % 256] = a * V[i + 33] + W[(i + 33) % 256];
          }
          if (i + 34 < N) {
            W[(i + 34) % 256] = a * V[i + 34] + W[(i + 34) % 256];
          }
          if (i + 35 < N) {
            W[(i + 35) % 256] = a * V[i + 35] + W[(i + 35) % 256];
          }
          if (i + 36 < N) {
            W[(i + 36) % 256] = a * V[i + 36] + W[(i + 36) % 256];
          }
          if (i + 37 < N) {
            W[(i + 37) % 256] = a * V[i + 37] + W[(i + 37) % 256];
          }
          if (i + 38 < N) {
            W[(i + 38) % 256] = a * V[i + 38] + W[(i + 38) % 256];
          }
          if (i + 39 < N) {
            W[(i + 39) % 256] = a * V[i + 39] + W[(i + 39) % 256];
          }
          if (i + 40 < N) {
            W[(i + 40) % 256] = a * V[i + 40] + W[(i + 40) % 256];
          }
          if (i + 41 < N) {
            W[(i + 41) % 256] = a * V[i + 41] + W[(i + 41) % 256];
          }
          if (i + 42 < N) {
            W[(i + 42) % 256] = a * V[i + 42] + W[(i + 42) % 256];
          }
          if (i + 43 < N) {
            W[(i + 43) % 256] = a * V[i + 43] + W[(i + 43) % 256];
          }
          if (i + 44 < N) {
            W[(i + 44) % 256] = a * V[i + 44] + W[(i + 44) % 256];
          }
          if (i + 45 < N) {
            W[(i + 45) % 256] = a * V[i + 45] + W[(i + 45) % 256];
          }
          if (i + 46 < N) {
            W[(i + 46) % 256] = a * V[i + 46] + W[(i + 46) % 256];
          }
          if (i + 47 < N) {
            W[(i + 47) % 256] = a * V[i + 47] + W[(i + 47) % 256];
          }
          if (i + 48 < N) {
            W[(i + 48) % 256] = a * V[i + 48] + W[(i + 48) % 256];
          }
          if (i + 49 < N) {
            W[(i + 49) % 256] = a * V[i + 49] + W[(i + 49) % 256];
          }
          if (i + 50 < N) {
            W[(i + 50) % 256] = a * V[i + 50] + W[(i + 50) % 256];
          }
          if (i + 51 < N) {
            W[(i + 51) % 256] = a * V[i + 51] + W[(i + 51) % 256];
          }
          if (i + 52 < N) {
            W[(i + 52) % 256] = a * V[i + 52] + W[(i + 52) % 256];
          }
          if (i + 53 < N) {
            W[(i + 53) % 256] = a * V[i + 53] + W[(i + 53) % 256];
          }
          if (i + 54 < N) {
            W[(i + 54) % 256] = a * V[i + 54] + W[(i + 54) % 256];
          }
          if (i + 55 < N) {
            W[(i + 55) % 256] = a * V[i + 55] + W[(i + 55) % 256];
          }
          if (i + 56 < N) {
            W[(i + 56) % 256] = a * V[i + 56] + W[(i + 56) % 256];
          }
          if (i + 57 < N) {
            W[(i + 57) % 256] = a * V[i + 57] + W[(i + 57) % 256];
          }
          if (i + 58 < N) {
            W[(i + 58) % 256] = a * V[i + 58] + W[(i + 58) % 256];
          }
          if (i + 59 < N) {
            W[(i + 59) % 256] = a * V[i + 59] + W[(i + 59) % 256];
          }
          if (i + 60 < N) {
            W[(i + 60) % 256] = a * V[i + 60] + W[(i + 60) % 256];
          }
          if (i + 61 < N) {
            W[(i + 61) % 256] = a * V[i + 61] + W[(i + 61) % 256];
          }
          if (i + 62 < N) {
            W[(i + 62) % 256] = a * V[i + 62] + W[(i + 62) % 256];
          }
          if (i + 63 < N) {
            W[(i + 63) % 256] = a * V[i + 63] + W[(i + 63) % 256];
          }
        }
        p_loop_action();
      }
      p_log << "UnrollManual \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollLoopPeeling(Index N = default_N) {
      // Do not change
      Real           a             = -1.0f;
      constexpr auto unroll_factor = 64;
      N                            = N % unroll_factor ? N : N + 1;
      auto rem                     = N % unroll_factor;

      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });
      std::vector<Real> W(256, 10);

      // --------
      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
#pragma omp simd
        for (Index i = 0; i < N - rem; i += unroll_factor) {
          W[(i + 0) % 256]  = a * V[i + 0] + W[(i + 0) % 256];
          W[(i + 1) % 256]  = a * V[i + 1] + W[(i + 1) % 256];
          W[(i + 2) % 256]  = a * V[i + 2] + W[(i + 2) % 256];
          W[(i + 3) % 256]  = a * V[i + 3] + W[(i + 3) % 256];
          W[(i + 4) % 256]  = a * V[i + 4] + W[(i + 4) % 256];
          W[(i + 5) % 256]  = a * V[i + 5] + W[(i + 5) % 256];
          W[(i + 6) % 256]  = a * V[i + 6] + W[(i + 6) % 256];
          W[(i + 7) % 256]  = a * V[i + 7] + W[(i + 7) % 256];
          W[(i + 8) % 256]  = a * V[i + 8] + W[(i + 8) % 256];
          W[(i + 9) % 256]  = a * V[i + 9] + W[(i + 9) % 256];
          W[(i + 10) % 256] = a * V[i + 10] + W[(i + 10) % 256];
          W[(i + 11) % 256] = a * V[i + 11] + W[(i + 11) % 256];
          W[(i + 12) % 256] = a * V[i + 12] + W[(i + 12) % 256];
          W[(i + 13) % 256] = a * V[i + 13] + W[(i + 13) % 256];
          W[(i + 14) % 256] = a * V[i + 14] + W[(i + 14) % 256];
          W[(i + 15) % 256] = a * V[i + 15] + W[(i + 15) % 256];
          W[(i + 16) % 256] = a * V[i + 16] + W[(i + 16) % 256];
          W[(i + 17) % 256] = a * V[i + 17] + W[(i + 17) % 256];
          W[(i + 18) % 256] = a * V[i + 18] + W[(i + 18) % 256];
          W[(i + 19) % 256] = a * V[i + 19] + W[(i + 19) % 256];
          W[(i + 20) % 256] = a * V[i + 20] + W[(i + 20) % 256];
          W[(i + 21) % 256] = a * V[i + 21] + W[(i + 21) % 256];
          W[(i + 22) % 256] = a * V[i + 22] + W[(i + 22) % 256];
          W[(i + 23) % 256] = a * V[i + 23] + W[(i + 23) % 256];
          W[(i + 24) % 256] = a * V[i + 24] + W[(i + 24) % 256];
          W[(i + 25) % 256] = a * V[i + 25] + W[(i + 25) % 256];
          W[(i + 26) % 256] = a * V[i + 26] + W[(i + 26) % 256];
          W[(i + 27) % 256] = a * V[i + 27] + W[(i + 27) % 256];
          W[(i + 28) % 256] = a * V[i + 28] + W[(i + 28) % 256];
          W[(i + 29) % 256] = a * V[i + 29] + W[(i + 29) % 256];
          W[(i + 30) % 256] = a * V[i + 30] + W[(i + 30) % 256];
          W[(i + 31) % 256] = a * V[i + 31] + W[(i + 31) % 256];
          W[(i + 32) % 256] = a * V[i + 32] + W[(i + 32) % 256];
          W[(i + 33) % 256] = a * V[i + 33] + W[(i + 33) % 256];
          W[(i + 34) % 256] = a * V[i + 34] + W[(i + 34) % 256];
          W[(i + 35) % 256] = a * V[i + 35] + W[(i + 35) % 256];
          W[(i + 36) % 256] = a * V[i + 36] + W[(i + 36) % 256];
          W[(i + 37) % 256] = a * V[i + 37] + W[(i + 37) % 256];
          W[(i + 38) % 256] = a * V[i + 38] + W[(i + 38) % 256];
          W[(i + 39) % 256] = a * V[i + 39] + W[(i + 39) % 256];
          W[(i + 40) % 256] = a * V[i + 40] + W[(i + 40) % 256];
          W[(i + 41) % 256] = a * V[i + 41] + W[(i + 41) % 256];
          W[(i + 42) % 256] = a * V[i + 42] + W[(i + 42) % 256];
          W[(i + 43) % 256] = a * V[i + 43] + W[(i + 43) % 256];
          W[(i + 44) % 256] = a * V[i + 44] + W[(i + 44) % 256];
          W[(i + 45) % 256] = a * V[i + 45] + W[(i + 45) % 256];
          W[(i + 46) % 256] = a * V[i + 46] + W[(i + 46) % 256];
          W[(i + 47) % 256] = a * V[i + 47] + W[(i + 47) % 256];
          W[(i + 48) % 256] = a * V[i + 48] + W[(i + 48) % 256];
          W[(i + 49) % 256] = a * V[i + 49] + W[(i + 49) % 256];
          W[(i + 50) % 256] = a * V[i + 50] + W[(i + 50) % 256];
          W[(i + 51) % 256] = a * V[i + 51] + W[(i + 51) % 256];
          W[(i + 52) % 256] = a * V[i + 52] + W[(i + 52) % 256];
          W[(i + 53) % 256] = a * V[i + 53] + W[(i + 53) % 256];
          W[(i + 54) % 256] = a * V[i + 54] + W[(i + 54) % 256];
          W[(i + 55) % 256] = a * V[i + 55] + W[(i + 55) % 256];
          W[(i + 56) % 256] = a * V[i + 56] + W[(i + 56) % 256];
          W[(i + 57) % 256] = a * V[i + 57] + W[(i + 57) % 256];
          W[(i + 58) % 256] = a * V[i + 58] + W[(i + 58) % 256];
          W[(i + 59) % 256] = a * V[i + 59] + W[(i + 59) % 256];
          W[(i + 60) % 256] = a * V[i + 60] + W[(i + 60) % 256];
          W[(i + 61) % 256] = a * V[i + 61] + W[(i + 61) % 256];
          W[(i + 62) % 256] = a * V[i + 62] + W[(i + 62) % 256];
          W[(i + 63) % 256] = a * V[i + 63] + W[(i + 63) % 256];
        }
#pragma omp simd
        for (Index i = N - rem; i < N; i++) {
          W[i % 256] = a * V[i] + W[i % 256];
        }
        p_loop_action();
      }
      p_log << "UnrollLoopPeeling\t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollLoopPeelingDirective(Index N = default_N) {
      // Do not change
      Real a                       = -1.0f;
      using batch                  = xsimd::batch<Real>;
      constexpr auto simd_width    = batch::size;
      constexpr auto w_size        = 256;
      constexpr auto unroll_factor = UNROLLFACTOR;
      static_assert(unroll_factor % simd_width == 0,
                    "Unroll factor not divisible by simd width");
      static_assert(w_size % unroll_factor == 0 && w_size % simd_width == 0,
                    "W size not divisible by unroll/simd width");
      N        = N % unroll_factor ? N : N + 1;
      auto rem = N % unroll_factor;

      auto V = std::views::iota(0, N) | std::views::transform([](int x) {
                 return static_cast<Real>(x);
               });

      std::array<Real, w_size> W;
      W.fill(10);
      batch a_vec(-1.0f);

      auto build_batch = [&]<Index... Is>(Index base_idx,
                                          std::index_sequence<Is...>) {
        return batch(V[base_idx + Is]...);
      };

      Index Nout = min(N, default_Nout);
      for (auto _ : p_loop_state) {
        for (Index i = 0; i < N - rem; i += unroll_factor) {
#pragma unroll
          for (Index j = 0; j < unroll_factor; j += simd_width) {
            batch v =
              build_batch(i + j, std::make_index_sequence<simd_width>{});
            batch w = xsimd::load_unaligned(&W[(i + j) % 256]);
            w       = a * v + w;
            w.store_unaligned(&W[(i + j) % 256]);
          }
        }
#pragma omp simd
        for (Index i = N - rem; i < N; i++) {
          W[i % 256] = a * V[i] + W[i % 256];
        }
        p_loop_action();
      }
      p_log << "UnrollLoopPeelingDirective\t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};

template<typename R, typename L>
array<char, 1> transform_LoopUnrolling_view<R, L>::p_one = {0};