#include "allocator_adaptor.hpp"
#include "numa_extensions.hpp"
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

namespace NUMA_omp {
  using namespace std;

  template<typename R = array<char, 1>, typename L = function<void(void)>>
  class Scan {
    private:
      static array<char, 1> p_one;
      stringstream          p_log;
      R                    &p_loop_state;
      L                     p_loop_action;

    public:
      Scan(
        R &loop_state = p_one, L loop_action = []() {})
          : p_loop_state(loop_state), p_loop_action(loop_action) {
        p_log << fixed << setprecision(2);
      }

      string get_log() { return p_log.str(); }

      using Index     = int;
      using Int       = int32_t;
      using Real      = float;
      using LongReal  = long double;
      using CReal     = complex<Real>;
      using LongCReal = complex<LongReal>;
      template<typename T>
      using Container =
        vector<T, numa::no_init_allocator<T>>; // TODO use no-init-allocator
                                               // this will cause old code crash
      // using Container = deque<T, allocator<T>>; // TODO use no-init-allocator
      template<typename T, int Fold = 1>
      using NUMAContainer = numa_extensions<Container<T>, Fold>;
      static constexpr auto stExec =
        execution::unseq;     // single-threaded execution policy
      static constexpr auto mtExec =
        execution::par_unseq; // multi-threaded execution policy

      static constexpr Index default_n    = 3;
      static constexpr Index default_m    = 2;
      static constexpr Index default_N    = 20;
      static constexpr Index default_Nout = 10;

      static constexpr Index default_nx = 3;
      static constexpr Index default_ny = 5;
      static constexpr Index default_nz = 3;
      static constexpr Index default_Nx = 4;
      static constexpr Index default_Ny = 8;
      static constexpr Index default_Nz = 2;

      auto exclusive_scan_scalar(Index N = default_N) { // exclusive scan scalar
        Index n    = default_n;
        Index m    = 1;
        Index Nout = min(N, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp parallel default(none) shared(V)
        {
          // #pragma GCC unroll
#pragma omp for simd
          for (size_t i = 0; i < V.size(); i++) {
            // V = [1, 1, 0, 1, 1, 0, 1, 1, 0, ...]
            V[i] = i % 3 <= 1;
          }
        }
        NUMAContainer<Int> W(N, -1);
        Int                init = 0;

        for (auto _ : p_loop_state) {
#pragma omp          parallel default(none) shared(V, W, init)
          {
            // #pragma GCC unroll
#pragma omp for simd reduction(inscan, + : init)
            for (size_t i = 0; i < V.size(); i++) {
              W[i] = init;
#pragma omp scan     exclusive(init)
              init += V[i];
            }
          }
          p_loop_action();
        }

        p_log << "Input:\t\t" << views::take(V, Nout) << '\n';
        p_log << "exclusive_scan:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto inclusive_scan_scalar(Index N = default_N) { // inclusive scan scalar
        Index n    = default_n;
        Index m    = 1;
        Index Nout = min(N, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp          parallel default(none) shared(V, n)
        {
          // #pragma GCC unroll
#pragma omp for simd
          for (size_t i = 0; i < V.size(); i++) { V[i] = i % n <= 1; }
        }
        NUMAContainer<Int> W(N, -1);
        Int                init = 0;

        for (auto _ : p_loop_state) {
#pragma omp          parallel default(none) shared(V, W, init)
          {
            // #pragma GCC unroll
#pragma omp for simd reduction(inscan, + : init)
            for (size_t i = 0; i < V.size(); i++) {
              init += V[i];
#pragma omp scan     inclusive(init)
              W[i] = init;
            }
          }
          p_loop_action();
        }

        p_log << "inclusive_scan:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto scan_scalar(
        Index N = default_N) { // universal scan scalar including init at front
                               // and reduction result at back
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp          parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = i % n <= 1; }
        }
        NUMAContainer<Int> W(Nsup, -1);
        Int                init = 0;

        W[0] = init;
        for (auto _ : p_loop_state) {
#pragma omp          parallel default(none) shared(V, W, init)
          {
#pragma omp for simd reduction(inscan, + : init)
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              init += V[i];
#pragma omp scan     inclusive(init)
              W[i + 1] = init;
            }
          }
          p_loop_action();
        }

        p_log << "scan:\t\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto scan_copy_if(
        Index N = default_N) { // scan used to implement copy_if aka pack
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp          parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          // V = [1, 11, 21, 1, 11, 21, 1, 11, 21, ...], n=3
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }
        NUMAContainer<Int> W(N, -1);

        NUMAContainer<Index> I(Nsup, -1);

        Index init = 0;
        Index sep  = 0;

        I[0] = init;
        for (auto _ : p_loop_state) {
          init = 0;
          I[0] = init;
#pragma omp          parallel default(none) shared(V, I, init)
          {
#pragma omp for simd reduction(inscan, + : init)
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              // flag = [1, 1, 0, 1, 1, 0, 1, 1, 0, ...]
              init += (V[i] <= 11);
#pragma omp scan     inclusive(init)
              I[i + 1] = init;
            }
          }
#pragma omp          parallel default(none) shared(W, I, V)
          {
#pragma omp for simd
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              if (I[i] < I[i + 1]) { W[I[i]] = V[i]; }
            }
          }
          p_loop_action();
        }

        sep = I[N];
        // W.erase(begin(W) + sep, end(W)); // prune W to length of copied
        // values
        p_log << "Input:\t\t" << views::take(V, Nout) << '\n';
        p_log << "scan copy_if I:\t" << views::take(I, Nout) << '\n';
        p_log << "scan copy_if W:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real), W.size() * sizeof(Real)};
      }

      auto scan_partition1(
        Index N = default_N) { // scan used to partition aka split
        Index           n    = default_n;
        constexpr Index m    = 1;
        Index           Nsup = N + 1;
        Index           Nout = min(Nsup, default_Nout);

        NUMAContainer<Real> V(N);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }
        NUMAContainer<Real> W(N, -1);     // max length N

        NUMAContainer<Index> I(Nsup, -1); // max length N
        Index                init = 0;
        Index                sep  = 0;

        I[0] = init;
        for (auto _ : p_loop_state) {
          init = 0;
          I[0] = init;
#pragma omp          parallel default(none) shared(V, I, init)
          {
#pragma omp for simd reduction(inscan, + : init)
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              init += (V[i] <= 11);
#pragma omp scan     inclusive(init)
              I[i + 1] = init;
            }
          }
          sep = I[N];
#pragma omp          parallel default(none) shared(W, I, V, sep)
          {
#pragma omp for simd
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              Index ind = (I[i] < I[i + 1]) ? I[i] : (i - I[i + 1]) + sep;
              W[ind]    = V[i];
            }
          }
          p_loop_action();
        }

        p_log << "scan partition I:\t" << views::take(I, Nout) << '\n';
        p_log << "scan partition W:\t" << sep << ':' << views::take(W, Nout)
              << '\n';
        return tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto scan_partition2(
        Index N = default_N) { // scan used to partition aka split
        Index           n    = default_n;
        constexpr Index m    = 2;
        Index           Nsup = N + 1;
        Index           Nout = min(Nsup, default_Nout);

        NUMAContainer<Real> V(N);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }
        using ElementTmp = tuple<Index, Index>;
        static_assert(tuple_size<ElementTmp>() == m);
        NUMAContainer<ElementTmp> I(Nsup, ElementTmp{-1, -1});
        NUMAContainer<Real>       W(N, -1); // max length N
        auto                      init = ElementTmp{0, 0};
        Index                     sep  = 0;
#pragma omp declare reduction(                                                 \
    + : ElementTmp : omp_out = ElementTmp{get<0>(omp_out) + get<0>(omp_in),    \
                                            get<1>(omp_out) + get<1>(omp_in)}) \
  initializer(omp_priv = ElementTmp{0, 0})
        I[0] = init;
        for (auto _ : p_loop_state) {
          init = ElementTmp{0, 0};
          I[0] = init;
#pragma omp          parallel default(none) shared(V, I, init)
          {
#pragma omp for simd reduction(inscan, + : init)
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              auto cond = (V[i] <= 11);
              init = ElementTmp{cond + get<0>(init), (!cond) + get<1>(init)};
#pragma omp scan     inclusive(init)
              I[i + 1] = init;
            }
          }
          sep = get<0>(I[N]);
#pragma omp          parallel default(none) shared(W, I, V, sep)
          {
#pragma omp for simd
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              auto [ia0, ia1] = I[i];
              auto [ib0, ib1] = I[i + 1];
              Index ind       = (ia0 < ib0) ? ia0 : ia1 + sep;
              W[ind]          = V[i];
            }
          }
          p_loop_action();
        }

        p_log << "scan partition I0:\t"
              << views::take(views::elements<0>(I), Nout) << '\n';
        p_log << "scan partition I1:\t"
              << views::take(views::elements<1>(I), Nout) << '\n';
        p_log << "scan partition W:\t" << sep << ':' << views::take(W, Nout)
              << '\n';
        return tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto scan_unique_copy(
        Index N = default_N) { // scan used to implement unique_copy
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Real> V(N);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }

        NUMAContainer<Index> I(Nsup, -1);
        NUMAContainer<Real>  W(N, -1); // max length N
        Index                init = 0;
        Index                sep  = 0;

        I[0] = init;
        for (auto _ : p_loop_state) {
          I[1] = I[0] + 1;
          init = 1;
#pragma omp          parallel default(none) shared(V, I, init)
          {
#pragma omp for simd reduction(inscan, + : init)
            // #pragma GCC unroll
            for (size_t i = 1; i < V.size(); i++) {
              init += ((V[i - 1] <= 11) != (V[i] <= 11));
#pragma omp scan     inclusive(init)
              I[i + 1] = init;
            }
          }
#pragma omp          parallel default(none) shared(W, I, V)
          {
#pragma omp for simd
            // #pragma GCC unroll
            for (size_t i = 0; i < V.size(); i++) {
              if (I[i] < I[i + 1]) { W[I[i]] = V[i]; }
            }
            p_loop_action();
          }
        }
        sep = I[N];
        // W.erase(begin(W) + sep, end(W)); // prune W to length of copied
        // values
        p_log << "scan unique I:\t" << views::take(I, Nout) << '\n';
        p_log << "scan unique W:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real), W.size() * sizeof(Real)};
      }
  };

  template<typename R, typename L> array<char, 1> Scan<R, L>::p_one = {0};
} // namespace NUMA_omp
