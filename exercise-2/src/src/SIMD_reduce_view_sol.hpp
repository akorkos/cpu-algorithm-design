
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
#include <vector>
#include <xsimd/xsimd.hpp>
#pragma once

using namespace std;

template<typename R = array<char, 1>, typename L = function<void(void)>>
class SIMD_reduce_view {
  private:
    static array<char, 1> p_one;
    stringstream          p_log;
    R                    &p_loop_state;
    L                     p_loop_action;

  public:
    SIMD_reduce_view(
      R &loop_state = p_one, L loop_action = []() {})
        : p_loop_state(loop_state), p_loop_action(loop_action) {
      p_log << setprecision(2);
    }

    string get_log() { return p_log.str(); }

    using Index = int;
    using Int   = int32_t;
    using Real  = float;
    using CReal = complex<Real>;
    template<typename T>
    using Container = vector<T, allocator<T>>; // TODO use no-init-allocator
    template<typename T> using View = std::ranges::repeat_view<T, Index>;

    template<typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec =
      execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec =
      execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n    = 3;
    static constexpr Index default_m    = 2;
    static constexpr Index default_N    = 24;
    static constexpr Index default_Nout = 10;

    auto benchReduceIterator(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO
      View<Real> V(10.f, N);
      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        for (auto it = V.begin(); it != V.end();) { sum += *it++; }
      }
      p_log << "Iterator \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceRange(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO

      View<Real> V(10.f, N);
      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        for (auto &c : V) { sum += c; }
      }
      p_log << "Range \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceStl(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO

      View<Real> V(10.f, N);
      for (auto _ : p_loop_state) {
        sum = std::reduce(V.begin(), V.end());
        // TODO
      }
      p_log << "Stl \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdStl(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO

      View<Real> V(10.f, N);
      for (auto _ : p_loop_state) {
        sum = std::reduce(std::execution::unseq, V.begin(), V.end());
        // TODO
      }
      p_log << "SimdStl \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdOmpIterator(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO
      View<Real> V(10.f, N);

      for (auto _ : p_loop_state) {
        sum = 0;
#pragma simd reduction(+ : sum)
        for (auto it = V.begin(); it != V.end(); it++) {
          // ?
          sum += *it;
        }
        // TODO
      }
      p_log << "OmpIterator \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdOmpRange(Index N = default_N) {
      // Do not change
      Real  sum;
      Index Nout = min(N, default_Nout);
      // TODO
      View<Real> V(10.f, N);

      for (auto _ : p_loop_state) {
        sum = 0;
// TODO
#pragma simd reduction(+ : sum)
        for (auto &c : V) { sum += c; }
      }
      p_log << "OmpRange \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXHorizontal(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Real           sum;
      Index          Nout = min(N, default_Nout);

      // TODO
      std::array<Real, 3 * simd_width> V;

      Index rem   = reinterpret_cast<long long>(&V) % 64;
      Index start = rem > 32 ? 0 : 33 - rem;
      std::fill(V.begin(), V.end(), 10);

      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        auto it = V.begin() + start;
        for (auto i = 0; i < N; i += simd_width) {
          batch simd_vec = batch::load_unaligned(&*it);
          sum += xsimd::reduce_add(simd_vec);
        }
      }
      p_log << "xsimd_Horizontal \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXVertical(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Real           sum;
      Index          Nout = min(N, default_Nout);

      // TODO
      batch                            sum_vec;
      std::array<Real, 3 * simd_width> V;
      std::fill(V.begin(), V.end(), 10);

      Index rem   = reinterpret_cast<long long>(&V) % 64;
      Index start = rem > 32 ? 0 : 33 - rem;

      start = (start + sizeof(Real) - 1) / sizeof(Real);
      // p_log << xsimd::is_aligned(&*(V.begin() + start)) << '\n';

      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        sum_vec = 0;
        auto it = V.begin() + start;
        for (auto i = 0; i < N; i += simd_width) {
          batch simd_vec = batch::load_unaligned(&*it);
          sum_vec += simd_vec;
        }
        sum = xsimd::reduce_add(sum_vec);
      }
      p_log << "xsimd_Vertical\t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXAlign(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Real           sum;
      Index          Nout = min(N, default_Nout);

      // TODO
      std::array<Real, 3 * simd_width> V;
      std::fill(V.begin(), V.end(), 10);

      Index rem   = reinterpret_cast<long long>(&V) % 64;
      Index start = rem < 32 ? 32 - rem : 64 - rem;

      start = (start + sizeof(Real) - 1) / sizeof(Real);
      // p_log << xsimd::is_aligned(&*(V.begin() + start)) << '\n';
      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        auto it = V.begin() + start;
        for (auto i = 0; i < N; i += simd_width) {
          batch simd_vec = batch::load_aligned(&*it);
          sum += xsimd::reduce_add(simd_vec);
        }
      }
      p_log << "xsimd_aligned \t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXAlignVertical(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Real           sum;
      Index          Nout = min(N, default_Nout);

      // TODO
      batch sum_vec;

      std::array<Real, 3 * simd_width> V;
      std::fill(V.begin(), V.end(), 10);

      Index rem   = reinterpret_cast<long long>(&V) % 64;
      Index start = rem < 32 ? 32 - rem : 64 - rem;

      start = (start + sizeof(Real) - 1) / sizeof(Real);
      // p_log << xsimd::is_aligned(&*(V.begin() + start)) << '\n';
      for (auto _ : p_loop_state) {
        sum = 0;
        // TODO
        auto it = V.begin() + start;
        for (auto i = 0; i < N; i += simd_width) {
          batch simd_vec = batch::load_aligned(&*it);
          sum_vec        = xsimd::add(simd_vec, sum_vec);
        }
        sum = xsimd::reduce_add(sum_vec);
      }
      p_log << "xsimd_aligned Vertical\t" << sum << '\n';
      return tuple{N * sizeof(Real), 0};
    }
};

template<typename R, typename L>
array<char, 1> SIMD_reduce_view<R, L>::p_one = {0};
