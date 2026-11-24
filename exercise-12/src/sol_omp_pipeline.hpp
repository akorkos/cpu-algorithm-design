#include "allocator_adaptor.hpp"
#include "numa_extensions.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <complex>
#include <cstdint>
#include <deque>
#include <execution>
#include <functional>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <ranges>
#include <vector>

namespace NUMA_omp {
  template<typename T>
  static void writePPM(T &buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
      // Map the iteration count to colors by just alternating between
      // two greys.
      char c = (buf[i] & 0x1) ? 240 : 20;
      for (int j = 0; j < 3; ++j) { fputc(c, fp); }
    }
    fclose(fp);
  }

#pragma omp declare simd

  template<typename T> T func(T x) {
    return x * x;
  }

#pragma omp declare simd

  template<typename T> T func2(T x) {
    return x * x + 2;
  }

  inline int32_t decompress(int32_t value, [[maybe_unused]] int32_t exponent) {
    return value;
  }

  inline int32_t compute(int32_t state, int32_t value) {
    return state ^ value;
  }

  inline int32_t required_exponent([[maybe_unused]] int32_t value) {
    return 0;
  }

  inline int32_t compress(int32_t value, [[maybe_unused]] int32_t exponent) {
    return value;
  }

  using namespace std;

  template<typename R = array<char, 1>, typename L = function<void(void)>>
  class Pipeline {
    private:
      static array<char, 1> p_one;
      stringstream          p_log;
      R                    &p_loop_state;
      L                     p_loop_action;

    public:
      Pipeline(
        R &loop_state = p_one, L loop_action = []() {})
          : p_loop_state(loop_state), p_loop_action(loop_action) {
        p_log << fixed << setprecision(2);
      }

      string get_log() { return p_log.str(); }

      template<typename T> struct triad_functor {
          const T a;

          triad_functor(T _a) : a{_a} {}

          T operator()(const T v, const T w) const { return a * v + w; }
      };

      using Index     = int;
      using Int       = int32_t;
      using Real      = float;
      using LongReal  = long double;
      using CReal     = complex<Real>;
      using LongCReal = complex<LongReal>;
      template<typename T>
      using Container = vector<T, numa::aligned_no_init_allocator<T, 4096>>;
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

      auto update_sequential(Index N = default_N) {
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real                a      = 2;
        constexpr int       stride = 1024;
        for (auto _ : p_loop_state) {
          // Z = a * V + W;
          {
            for (size_t i = 0; i < V.size() - stride; i++) {
              V[i + stride] = func(V[i]);
            }
          }
          p_loop_action();
        }

        // output
        std::copy_n(cbegin(V), default_Nout,
                    std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto update_parallel_simd(
        Index N = default_N) { // transform: fused operations in one lambda,
                               // mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real                a      = 2;
        constexpr int       stride = 1024;

        for (auto _ : p_loop_state) {
#pragma simd
#pragma unroll
          for (size_t j = 0; j < stride; j++) {
            for (size_t i = 0; i < V.size(); i += stride) {
              V[i + j] = func(V[i]);
            }
          }
          p_loop_action();
        }
        // output
        std::copy_n(cbegin(V), default_Nout,
                    std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto update_parallel_nested_loop(
        Index N = default_N) { // transform: fused operations in one lambda,
                               // mapping indices to values
        NUMAContainer<Real, 262'144> V(itSeq(0), itSeq(N));
        NUMAContainer<Real, 262'144> W(N, 3);
        Real                         a      = 2;
        constexpr int                stride = 4096;

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, W, a)
          {
#pragma omp for simd
#pragma unroll
            for (size_t j = 0; j < stride; ++j) {
              for (size_t i = j; i + stride < V.size(); i += stride) {
                V[i + stride] = func(V[i]);
              }
            }
          }
          p_loop_action();
        }

        // output
        std::copy_n(cbegin(V), default_Nout,
                    std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto complex_pipeline(Index N = default_N) {
        constexpr size_t page_size = 4096;
        constexpr size_t stride    = page_size / sizeof(Real); // 1024 values
        constexpr size_t blocks_per_page    = 4;
        constexpr size_t threads_per_region = 16;
        constexpr size_t chains_per_block   = stride / blocks_per_page; // 256
        constexpr size_t chains_per_thread =
          chains_per_block / threads_per_region;                        // 16

        // Number of consecutive source pages processed before the chains are
        // handed to the next NUMA region. This is the pipeline tuning
        // parameter.
        constexpr size_t pages_per_segment = 64;

        static_assert(sizeof(Real) == 4);
        static_assert(stride == 1024);
        static_assert(chains_per_thread * sizeof(Real) == 64);

        assert(N > 0);

        const size_t value_count = static_cast<size_t>(N);

        assert(value_count % stride == 0);

        const size_t page_count = value_count / stride;

        assert(page_count > 1);

        const int region_count = omp_get_num_places();

        // The schedule below has one pipeline stage for each of the four
        // independent blocks in a page and one worker team per NUMA region.
        assert(region_count == static_cast<int>(blocks_per_page));
        assert(page_count %
                 (static_cast<size_t>(region_count) * pages_per_segment) ==
               0);

        omp_set_dynamic(0);
        omp_set_max_active_levels(2);

        /*
         * The count-only constructor leaves the trivially constructible Real
         * values untouched. The following cyclic first touch assigns P pages
         * to region 0, the next P pages to region 1, and so on.
         */
        NUMAContainer<Real> V(value_count);

        const size_t allocation_segment_count =
          (page_count + pages_per_segment - 1) / pages_per_segment;

#pragma omp parallel num_threads(region_count) proc_bind(spread) default(none) \
  shared(V) firstprivate(region_count, page_count, allocation_segment_count)
        {
          const int region = omp_get_place_num();

          assert(region >= 0 && region < region_count);

          for (size_t segment = static_cast<size_t>(region);
               segment < allocation_segment_count;
               segment += static_cast<size_t>(region_count)) {
            const size_t first_page = segment * pages_per_segment;
            const size_t last_page =
              std::min(first_page + pages_per_segment, page_count);

            for (size_t page = first_page; page < last_page; ++page) {
              Real *const page_values = V.data() + page * stride;

#pragma omp simd     aligned(page_values : page_size)
              for (size_t chain = 0; chain < stride; ++chain) {
                page_values[chain] = static_cast<Real>(page * stride + chain);
              }
            }
          }
        }

        assert(reinterpret_cast<uintptr_t>(V.data()) % page_size == 0);

        struct alignas(64) CompletionToken {
            std::<ptrdiff_t>segment{-1};
        };

        std::array<CompletionToken, blocks_per_page> completed;

        // There is no update starting at the final page because its successor
        // would be outside V.
        const size_t source_page_count = page_count - 1;
        const size_t computation_segment_count =
          (source_page_count + pages_per_segment - 1) / pages_per_segment;

        for (auto _ : p_loop_state) {
          for (auto &token : completed) {
            token.segment.store(-1, std::memory_order_relaxed);
          }

          /*
           * The outer threads are the four NUMA-region pipeline stages.
           * Each outer thread creates one persistent 16-thread local team.
           */
#pragma omp parallel num_threads(region_count) proc_bind(spread) default(none) \
  shared(V, completed)                                                         \
  firstprivate(region_count, source_page_count, computation_segment_count)
          {
            const int region = omp_get_place_num();

            assert(region >= 0 && region < region_count);

#pragma omp parallel num_threads(threads_per_region)                           \
  proc_bind(close) default(none) shared(V, completed) firstprivate(            \
      region, region_count, source_page_count, computation_segment_count)
            {
              const size_t team_thread =
                static_cast<size_t>(omp_get_thread_num());

              assert(omp_get_num_threads() ==
                     static_cast<int>(threads_per_region));

              /*
               * local_step is wave - region. For R == B == 4:
               *
               *   block   = (wave - region) % 4
               *   segment = ((wave - region) / 4) * 4 + region
               *
               * Advancing local_step independently in each region gives
               * the same wavefront without a global barrier per wave.
               */
              for (size_t local_step = 0;; ++local_step) {
                const size_t block = local_step % blocks_per_page;
                const size_t round = local_step / blocks_per_page;
                const size_t segment =
                  round * static_cast<size_t>(region_count) +
                  static_cast<size_t>(region);

                if (segment >= computation_segment_count) { break; }

                const ptrdiff_t predecessor =
                  static_cast<ptrdiff_t>(segment) - 1;

                // Only the team leader polls the cross-region token.
                if (team_thread == 0) {
                  while (completed[block].segment.load(
                           std::memory_order_acquire) < predecessor) {
                    _mm_pause();
                  }
                }

                // Release the other 15 threads only when this segment's
                // input boundary has been produced.
#pragma omp barrier

                const size_t first_page = segment * pages_per_segment;
                const size_t last_page =
                  std::min(first_page + pages_per_segment, source_page_count);
                const size_t first_chain =
                  block * chains_per_block + team_thread * chains_per_thread;

                for (size_t page = first_page; page < last_page; ++page) {
                  Real *const source_page = V.data() + page * stride;

                  /*
                   * The first block brings all 64 cache lines of the
                   * page toward the cache. Each team thread issues
                   * four of the prefetches.
                   */
                  if (block == 0) {
                    constexpr size_t values_per_cache_line = 64 / sizeof(Real);
                    constexpr size_t cache_lines_per_page  = page_size / 64;

                    for (size_t cache_line = team_thread;
                         cache_line < cache_lines_per_page;
                         cache_line += threads_per_region) {
                      __builtin_prefetch(
                        source_page + cache_line * values_per_cache_line, 0, 3);
                    }
                  }

                  Real *const src = source_page + first_chain;
                  Real *const dst = src + stride;

#pragma omp simd aligned(src, dst : 64)
                  for (size_t lane = 0; lane < chains_per_thread; ++lane) {
                    dst[lane] = func(src[lane]);
                  }
                }

                // All 16 cache-line groups must be complete before the
                // next NUMA region may continue this block's chains.
#pragma omp barrier

                if (team_thread == 0) {
                  completed[block].segment.store(
                    static_cast<ptrdiff_t>(segment), std::memory_order_release);
                }

                // Keep this local team together when it rotates to the
                // next independent block.
#pragma omp barrier
              }
            }
          }

          p_loop_action();
        }

        std::copy_n(cbegin(V), default_Nout,
                    std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';

        return std::tuple{value_count * sizeof(Real),
                          value_count * sizeof(Real)};
      }

      auto Block_compress_pipeline(Index N = default_N) {
        constexpr size_t simd_length      = 8;
        constexpr size_t threads_per_team = 8;
        constexpr size_t values_per_block =
          threads_per_team * simd_length; // 64

        // Amortizes the two team barriers over several blocks.
        constexpr size_t blocks_per_batch = 64;

        static_assert(values_per_block == 64);

        assert(N > 0);
        assert(static_cast<size_t>(N) % values_per_block == 0);

        const size_t value_count = static_cast<size_t>(N);
        const size_t block_count = value_count / values_per_block;

        NUMAContainer<Int> values(itSeq(Int{0}), itSeq(static_cast<Int>(N)));
        NUMAContainer<Int> exponents(block_count, Int{0});

        const int block_team_count = omp_get_num_places();

        assert(block_team_count > 0);
        assert(block_count % static_cast<size_t>(block_team_count) == 0);

        for (auto _ : p_loop_state) {
          // One persistent block team per NUMA region.
#pragma omp parallel num_threads(block_team_count) proc_bind(spread)
          {
            const int place = omp_get_place_num();

            // This matches the contiguous first-touch distribution used by
            // NUMAContainer with Fold == 1.
            const size_t first_block = block_count *
                                       static_cast<size_t>(place) /
                                       static_cast<size_t>(block_team_count);

            const size_t last_block = block_count *
                                      static_cast<size_t>(place + 1) /
                                      static_cast<size_t>(block_team_count);

            // One candidate exponent from each of the eight threads.
            alignas(64)
              std::array<std::array<Int, blocks_per_batch>, threads_per_team>
                exponent_candidates{};

#pragma omp parallel num_threads(threads_per_team) proc_bind(close)            \
  shared(values, exponents, exponent_candidates)
            {
              const size_t team_thread =
                static_cast<size_t>(omp_get_thread_num());

              // Thread t always owns values [t*8, t*8+8) in every block.
              alignas(32) std::array<Int, simd_length> chain_state{};

              for (size_t batch_begin = first_block; batch_begin < last_block;
                   batch_begin += blocks_per_batch) {
                const size_t batch_count =
                  std::min(blocks_per_batch, last_block - batch_begin);

                /*
                 * Stage 1: decompress.
                 *
                 * Each thread decompresses its eight consecutive values
                 * from every block in the batch.
                 */
                for (size_t slot = 0; slot < batch_count; ++slot) {
                  const size_t block          = batch_begin + slot;
                  const Int    input_exponent = exponents[block];

                  Int *block_values = values.data() + block * values_per_block +
                                      team_thread * simd_length;

#pragma omp simd simdlen(8) aligned(block_values : 32)
                  for (size_t lane = 0; lane < simd_length; ++lane) {
                    block_values[lane] =
                      decompress(block_values[lane], input_exponent);
                  }
                }

                /*
                 * Stage 2: chained computation.
                 *
                 * At the same time, each thread determines the maximum
                 * exponent required by its eight values.
                 */
                for (size_t slot = 0; slot < batch_count; ++slot) {
                  const size_t block = batch_begin + slot;

                  Int *block_values = values.data() + block * values_per_block +
                                      team_thread * simd_length;

                  Int local_exponent = std::numeric_limits<Int>::lowest();

#pragma omp simd simdlen(8) aligned(block_values : 32)                         \
  reduction(max : local_exponent)
                  for (size_t lane = 0; lane < simd_length; ++lane) {
                    const Int result =
                      compute(chain_state[lane], block_values[lane]);

                    chain_state[lane]  = result;
                    block_values[lane] = result;

                    local_exponent =
                      std::max(local_exponent, required_exponent(result));
                  }

                  exponent_candidates[team_thread][slot] = local_exponent;
                }

                // All eight candidates must be available.
#pragma omp barrier

                /*
                 * Exponent reduction.
                 *
                 * The eight threads distribute the batch's block
                 * reductions among themselves.
                 */
                for (size_t slot = team_thread; slot < batch_count;
                     slot += threads_per_team) {
                  Int block_exponent = std::numeric_limits<Int>::lowest();

#pragma omp simd simdlen(8) reduction(max : block_exponent)
                  for (size_t contributor = 0; contributor < threads_per_team;
                       ++contributor) {
                    block_exponent = std::max(
                      block_exponent, exponent_candidates[contributor][slot]);
                  }

                  exponents[batch_begin + slot] = block_exponent;
                }

                // Compression cannot begin until all block exponents
                // have been written.
#pragma omp barrier

                /*
                 * Stage 3: compress.
                 */
                for (size_t slot = 0; slot < batch_count; ++slot) {
                  const size_t block           = batch_begin + slot;
                  const Int    output_exponent = exponents[block];

                  Int *block_values = values.data() + block * values_per_block +
                                      team_thread * simd_length;

#pragma omp simd simdlen(8) aligned(block_values : 32)
                  for (size_t lane = 0; lane < simd_length; ++lane) {
                    block_values[lane] =
                      compress(block_values[lane], output_exponent);
                  }
                }

                // The next batch's first barrier also ensures every
                // thread has finished compressing this batch.
              }
            }
          }

          p_loop_action();
        }

        std::copy_n(cbegin(values), default_Nout,
                    std::ostream_iterator<Int>(p_log, ","));

        p_log << '\n';

        const size_t representation_bytes =
          value_count * sizeof(Int) + block_count * sizeof(Int);

        return std::tuple{representation_bytes, representation_bytes};
      }
  };

  template<typename R, typename L> array<char, 1> Pipeline<R, L>::p_one = {0};
} // namespace NUMA_omp
