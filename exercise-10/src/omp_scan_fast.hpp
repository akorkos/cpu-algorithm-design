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

      auto scan_twopass_helper(NUMAContainer<Real> &V, Index N) {
        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index               blocknum = nodes * blocks_per_node;
        NUMAContainer<Real> block_sums(nodes * blocks_per_node + 1, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
        {
          // #pragma GCC unroll
#pragma omp for
          for (size_t i = 0; i < blocknum; i++) {
            Real   local_block_sum = 0;
            size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
            for (size_t j = 0; j < block_size; j++) {
              local_block_sum += V[block_start + j];
            }
            block_sums[i + 1] = local_block_sum;
          }
        }

        std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                            block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
        {
          // #pragma GCC unroll
#pragma omp for
          for (size_t i = 0; i < blocknum; i++) {
            Real   local_block_sum = block_sums[i];
            size_t block_start     = i * block_size;
            std::inclusive_scan(stExec, V.begin() + block_start,
                                V.begin() + block_start + block_size,
                                V.begin() + block_start, std::plus<>{},
                                local_block_sum);
          }
        }

        return V;
      }

      auto scan_twopass(Index N = default_N) {
        Index n    = default_n;
        Index m    = 1;
        Index Nout = min(N, default_Nout);
        Nout       = 32;

        NUMAContainer<Real> V(itSeq(0), itSeq(N));

        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index               blocknum = nodes * blocks_per_node;
        NUMAContainer<Real> block_sums(nodes * blocks_per_node + 1, 0);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Real   local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += V[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Real   local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::inclusive_scan(stExec, V.begin() + block_start,
                                  V.begin() + block_start + block_size,
                                  V.begin() + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
        }
        p_log << "Input:\t\t" << views::take(V, Nout) << '\n';
        p_log << "inclusive_scan:\t" << views::take(V, Nout) << '\n';
        return tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto exclusive_scan_scalar(Index N = default_N) { // exclusive scan scalar
        Index n    = default_n;
        Index m    = 1;
        Index Nout = min(N, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp parallel default(none) shared(V)
        {
          // #pragma GCC unroll
#pragma omp for simd
          for (size_t i = 0; i < V.size(); i++) { V[i] = i % 3 <= 1; }
        }
        // NUMAContainer<Real> V(itSeq(0), itSeq(N));
        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index               blocknum = nodes * blocks_per_node;
        NUMAContainer<Real> block_sums(nodes * blocks_per_node + 1, 0);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Real   local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += V[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Real   local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::exclusive_scan(stExec, V.begin() + block_start,
                                  V.begin() + block_start + block_size,
                                  V.begin() + block_start, local_block_sum);
            }
          }
        }

        p_log << "Input:\t\t" << views::take(V, Nout) << '\n';
        p_log << "exclusive_scan:\t" << views::take(V, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto inclusive_scan_scalar(Index N = default_N) { // inclusive scan scalar
        Index n    = default_n;
        Index m    = 1;
        Index Nout = min(N, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp parallel default(none) shared(V, n)
        {
          // #pragma GCC unroll
#pragma omp for simd
          for (size_t i = 0; i < V.size(); i++) { V[i] = i % n <= 1; }
        }
        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index              blocknum = nodes * blocks_per_node;
        NUMAContainer<Int> block_sums(nodes * blocks_per_node + 1, 0);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += V[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::inclusive_scan(stExec, V.begin() + block_start,
                                  V.begin() + block_start + block_size,
                                  V.begin() + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
        }

        p_log << "inclusive_scan:\t" << views::take(V, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto scan_scalar(
        Index N = default_N) { // universal scan scalar including init at front
                               // and reduction result at back
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Int> V(Nsup);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i + 1] = i % n <= 1; }
        }
        // NUMAContainer<Int> W(Nsup, -1);
        Int init = 0;

        V[0] = init;

        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size() - 1) / block_size) / nodes);
        Index              blocknum = nodes * blocks_per_node;
        NUMAContainer<Int> block_sums(nodes * blocks_per_node + 1, 0);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = 0;
              size_t block_start     = i * block_size + 1;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += V[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none) shared(V, block_size, block_sums, blocknum)
          {
            // #pragma GCC unroll
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = block_sums[i];
              size_t block_start     = i * block_size + 1;
              std::inclusive_scan(stExec, V.begin() + block_start,
                                  V.begin() + block_start + block_size,
                                  V.begin() + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
        }

        p_log << "scan:\t\t" << views::take(V, Nout) << '\n';
        return tuple{N * sizeof(Int), N * sizeof(Int)};
      }

      auto scan_copy_if(
        Index N = default_N) { // scan used to implement copy_if aka pack
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Int> V(N);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }

        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index              blocknum = nodes * blocks_per_node;
        NUMAContainer<Int> block_sums(nodes * blocks_per_node + 1, 0);
        NUMAContainer<Int> W(N, -1);

        NUMAContainer<Index> I(Nsup, -1);

        Index init  = 0;
        Index sep   = 0;
        auto  Vview = V | views::transform([](auto i) { return i <= 11; });
        I[0]        = init;
        for (auto _ : p_loop_state) {
          init = 0;
          I[0] = init;

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += Vview[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum, I)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::inclusive_scan(stExec, Vview.begin() + block_start,
                                  Vview.begin() + block_start + block_size,
                                  I.begin() + 1 + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
#pragma omp parallel default(none) shared(W, I, V)
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

        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size()) / block_size) / nodes);
        Index               blocknum = nodes * blocks_per_node;
        NUMAContainer<Int>  block_sums(nodes * blocks_per_node + 1, 0);
        NUMAContainer<Real> W(N, -1);     // max length N

        NUMAContainer<Index> I(Nsup, -1); // max length N
        Index                init = 0;
        Index                sep  = 0;
        auto Vview = V | views::transform([](auto i) { return i <= 11; });
        I[0]       = init;
        for (auto _ : p_loop_state) {
          init = 0;
          I[0] = init;

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += Vview[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, 0);

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum, I)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::inclusive_scan(stExec, Vview.begin() + block_start,
                                  Vview.begin() + block_start + block_size,
                                  I.begin() + 1 + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
          sep = I[N];
#pragma omp parallel default(none) shared(W, I, V, sep)
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

      auto scan_unique_copy(
        Index N = default_N) { // scan used to implement unique_copy
        Index n    = default_n;
        Index m    = 1;
        Index Nsup = N + 1;
        Index Nout = min(Nsup, default_Nout);

        NUMAContainer<Real> V(N + 1);
#pragma omp parallel default(none) shared(V, n)
        {
#pragma omp for simd
          // #pragma GCC unroll
          for (size_t i = 0; i < V.size(); i++) { V[i] = 1 + 10 * (i % n); }
        }

        const int nodes      = omp_get_num_places();
        const int block_size = (1 << 16);
        // NUMAContainer<Real> W(itSeq(0), itSeq(N));
        const int blocks_per_node =
          std::ceil((static_cast<double>(V.size() - 1) / block_size) / nodes);
        Index                blocknum = nodes * blocks_per_node;
        NUMAContainer<Int>   block_sums(nodes * blocks_per_node + 1, 0);
        NUMAContainer<Index> I(Nsup + 1, -1);
        NUMAContainer<Real>  W(N, -1); // max length N
        Index                init = 0;
        Index                sep  = 0;

        auto Vview = ranges::views::iota(1, N) | views::transform([&V](int i) {
                       return ((V[i - 1] <= 11) != (V[i] <= 11));
                     });
        I[0]       = init;
        for (auto _ : p_loop_state) {
          I[1] = I[0] + 1;

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = 0;
              size_t block_start     = i * block_size;
#pragma omp simd reduction(+ : local_block_sum)
              for (size_t j = 0; j < block_size; j++) {
                local_block_sum += Vview[block_start + j];
              }
              block_sums[i + 1] = local_block_sum;
            }
          }

          std::inclusive_scan(stExec, block_sums.begin(), block_sums.end(),
                              block_sums.begin(), std::plus<>{}, I[1]);

#pragma omp parallel default(none)                                             \
  shared(Vview, block_size, block_sums, blocknum, I)
          {
#pragma omp for
            for (size_t i = 0; i < blocknum; i++) {
              Int    local_block_sum = block_sums[i];
              size_t block_start     = i * block_size;
              std::inclusive_scan(stExec, Vview.begin() + block_start,
                                  Vview.begin() + block_start + block_size,
                                  I.begin() + 2 + block_start, std::plus<>{},
                                  local_block_sum);
            }
          }
#pragma omp parallel default(none) shared(W, I, V)
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
        p_log << "scan unique I:\t" << views::take(I, Nout) << '\n';
        p_log << "scan unique W:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real), W.size() * sizeof(Real)};
      }

      auto reduce_scalar_norm2(Index N = default_N) { // reduce scalar norm2
        Index n    = 1;
        Index m    = n;
        Index Nout = min(N, default_Nout);

        NUMAContainer<Real> V(N, 1);
        LongReal            init = 0;
        LongReal            res  = 0;

        for (auto _ : p_loop_state) {
#pragma omp          parallel default(none) shared(res, V)
          {
#pragma omp for simd reduction(+ : res)
            for (size_t i = 0; i < V.size(); i++) { res += V[i] * V[i]; }
          }
          p_loop_action();
        }
        res = sqrt(res);

        p_log << "norm2:\t" << res << '\n';
        return tuple{N * sizeof(Real), 0};
      }

      auto reduce_n1(Index N = default_N) { // reduce n(N) -> 1
        constexpr Index n    = 3;
        Index           m    = 1;
        Index           Nout = min(N, default_Nout);

        NUMAContainer<Real> V0(N, 1);
        NUMAContainer<Real> V1(N, 1);
        NUMAContainer<Real> V2(N, 1);
        auto                V = views::zip(V0, V1, V2);

        using ElementV = ranges::range_value_t<decltype(V)>;
        static_assert(tuple_size<ElementV>() == n);
#pragma omp          parallel default(none) shared(V)
        {
#pragma omp for simd
          for (size_t i = 0; i < V.size(); i++) {
            V[i] = ElementV{1 * i, 2 * i, 3 * i};
          }
        }
        LongReal init = 0;
        LongReal res  = 0;

        for (auto _ : p_loop_state) {
#pragma omp          parallel default(none) shared(res, V)
          {
#pragma omp for simd reduction(+ : res)
            for (size_t i = 0; i < V.size(); i++) {
              auto [v0, v1, v2] = V[i];
              // if i % 3 == 0 the v0;
              // if i % 3 == 1 the v1;
              // if i % 3 == 2 the v2;
              res += (i % 3 > 0) ? ((i % 3 == 2) ? v2 : v1) : v0;
            }
          }
          p_loop_action();
        }

        p_log << "select:\t" << res << '\n';
        return tuple{N * sizeof(ElementV), 0};
      }

      auto reduce_1m(Index N = default_N) { // reduce 1(N) -> m
        Index           n    = 1;
        constexpr Index m    = 3;
        Index           Nout = min(N, default_Nout);

        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        using ElementW = tuple<Real, Real, LongReal>;
        static_assert(tuple_size<ElementW>() == m);
        ElementW init{numeric_limits<Real>::lowest(),
                      numeric_limits<Real>::max(), 0}; // lowest() != min()
        ElementW res{0, 0, 0};

#pragma omp declare reduction(                                                 \
    elementreduction:ElementW : omp_out =                                      \
      ElementW{max(get<0>(omp_in), get<0>(omp_out)),                           \
                 min(get<1>(omp_in), get<1>(omp_out)),                         \
                 (get<2>(omp_in) + get<2>(omp_out))})                          \
  initializer(omp_priv = ElementW{numeric_limits<Real>::lowest(),              \
                                    numeric_limits<Real>::max(), 0})

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(res, V)
          {
#pragma omp for reduction(elementreduction : res)
            for (size_t i = 0; i < V.size(); i++) {
              auto tempv = ElementW{V[i], V[i], V[i]};
              res        = ElementW{max(get<0>(tempv), get<0>(res)),
                                    min(get<1>(tempv), get<1>(res)),
                                    (get<2>(tempv) + get<2>(res))};
            }
          }
          p_loop_action();
        }

        p_log << "max,min,+:\t" << res << '\n';
        return tuple{N * sizeof(Real), 0};
      }
  };

  template<typename R, typename L> array<char, 1> Scan<R, L>::p_one = {0};
} // namespace NUMA_omp
