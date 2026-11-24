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
#include <random>
#include <ranges>
#include <vector>
// #include <omp.h>

#include <xsimd/xsimd.hpp>

#pragma once

using namespace std;

bool touches_single_cache_line(const void *ptr, std::size_t size) {
  constexpr std::size_t CACHE_LINE_SIZE = 64;
  uintptr_t             start           = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t             end             = start + size - 1;

  std::size_t start_line = start / CACHE_LINE_SIZE;
  std::size_t end_line   = end / CACHE_LINE_SIZE;

  return start_line == end_line;
}

template<typename R = array<char, 1>, typename L = function<void(void)>>
class SIMD_transform_view {
  private:
    static array<char, 1> p_one;
    stringstream          p_log;
    R                    &p_loop_state;
    L                     p_loop_action;

  public:
    SIMD_transform_view(
      R &loop_state = p_one, L loop_action = []() {})
        : p_loop_state(loop_state), p_loop_action(loop_action) {
      p_log << setprecision(2);
    }

    string get_log() { return p_log.str(); }

    using Index = int;
    using Int   = int32_t;
    using Real  = float;
    using CReal = complex<Real>;

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
    static constexpr Index default_Nout = 10;

    auto benchTransformIterator(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO

      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      auto Wview = views::iota(0, N) | views::transform([&](auto i) -> Real & {
                     return W[i % container_size];
                   });

      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
        for (auto v = V.begin(); v != V.end(); ++v, ++w) { *w = a * (*v) + *w; }
        p_loop_action();
      }

      p_log << "Iterator \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);

      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
        for (auto v = V.begin(); v != V.end();
             v += simd_width, w += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          for (Index i = 0; i < simd_width; ++i) {
            *(w + i) = a * (*(v + i)) + *(w + i);
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop_Index(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      // auto Wview = views::iota(0) | views::take(N) |
      // views::transform([&](auto i) -> Real &
      //                                                                 {
      //                                                                 return
      //                                                                 W[i %
      //                                                                 container_size];
      //                                                                 });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = W.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          for (Index i = 0; i < simd_width; ++i) {
            *(w + (j + i) % container_size) =
              a * (*(v + j + i)) + *(w + (j + i) % container_size);
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop_Index_view(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          for (Index i = 0; i < simd_width; ++i) {
            *(w + (j + i)) = a * (*(v + j + i)) + *(w + (j + i));
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop_Index_Once(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      // auto Wview = views::iota(0) | views::take(N) |
      // views::transform([&](auto i) -> Real &
      //                                                                 {
      //                                                                 return
      //                                                                 W[i %
      //                                                                 container_size];
      //                                                                 });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = W.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          auto local_index = j % container_size;
          for (Index i = 0; i < simd_width; ++i) {
            *(w + local_index + i) =
              a * (*(v + j + i)) + *(w + local_index + i);
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop_Index_Mask(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      constexpr unsigned int mask = 0xFF;
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = W.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          auto local_index = j & mask;
          for (Index i = 0; i < simd_width; ++i) {
            *(w + local_index + i) =
              a * (*(v + j + i)) + *(w + local_index + i);
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop_view_Mask(Index N = default_N) {
      // Do not change
      Real            a          = -1.0f;
      constexpr Index simd_width = 8;
      Index           Nout       = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      constexpr unsigned int mask = 0xFF;
      auto                   Wview =
        views::iota(0) | views::take(N) |
        views::transform([&](auto i) -> Real & { return W[i & mask]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
          // Manually unrolled loop body — emulates SIMD parallel execution
          // auto local_index = j & mask;
          for (Index i = 0; i < simd_width; ++i) {
            *(w + j + i) = a * (*(v + j + i)) + *(w + j + i);
          }
        }
        p_loop_action();
      }

      p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRange(Index N = default_N) {
      // Do not change
      constexpr Index simd_width = 8;
      Real            a          = -1.0f;
      Index           Nout       = min(N, default_Nout);

      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      for (auto _ : p_loop_state) {
        // TODO
        for (auto i : ranges::iota_view(0, N)) {
          W[i % container_size] = a * V[i] + W[i % container_size];
        }
        p_loop_action();
      }

      p_log << "Range \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRangeInnerLoop(Index N = default_N) {
      // Do not change
      constexpr Index simd_width = 8;
      Real            a          = -1.0f;
      Index           Nout       = min(N, default_Nout);

      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto in = ranges::transform_view(
        ranges::iota_view(0, N / simd_width),
        [simd_width](Index s) { return s * simd_width; });

      for (auto _ : p_loop_state) {
        // TODO
        for (auto u : in) {
          for (Index i = 0; i < simd_width; ++i) {
            W[(u + i) % container_size] =
              a * V[u + i] + W[(u + i) % container_size];
          }
        }
        p_loop_action();
      }

      p_log << "RangeInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformStl(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      constexpr Index simd_width = 8;
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        std::transform(V.begin(), V.end(), Wview.begin(), Wview.begin(),
                       [a](const auto v, const auto w) { return a * v + w; });
        p_loop_action();
      }

      p_log << "Stl \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformSimdStl(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      constexpr Index simd_width = 8;
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        std::transform(std::execution::unseq, V.begin(), V.end(), Wview.begin(),
                       Wview.begin(),
                       [a](const auto v, const auto w) { return a * v + w; });
        p_loop_action();
      }

      p_log << "SimdStl \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransform(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Index          Nout       = min(N, default_Nout);
      // TODO
      std::array<Real, 3 * simd_width> V;
      std::array<Real, 3 * simd_width> W;

      Index remV   = reinterpret_cast<long long>(&V) % 64;
      Index remW   = reinterpret_cast<long long>(&W) % 64;
      Index startV = remV > 32 ? 0 : 33 - remV;
      Index startW = remW > 32 ? 0 : 33 - remW;

      startW = (startW + sizeof(Real) - 1) / sizeof(Real);
      startV = (startV + sizeof(Real) - 1) / sizeof(Real);
      // p_log << xsimd::is_aligned(&*(V.begin() + startV)) << '\n';
      // p_log << xsimd::is_aligned(&*(W.begin() + startW)) << '\n';
      // p_log << touches_single_cache_line(&*(W.begin() + startW), 32) << '\n';
      // p_log << touches_single_cache_line(&*(V.begin() + startV), 32) << '\n';
      std::iota(V.begin() + startV, V.end(), Real{1});
      std::iota(W.begin() + startW, W.end(), Real{1});
      for (auto _ : p_loop_state) {
        // using xsimd with aligned container
        batch a_vec(-1.0f);
        // TODO
        auto v = V.begin() + startV;
        auto w = W.begin() + startW;
        for (Index i = 0; i < N / simd_width; i++) {
          batch x_vec = batch::load_unaligned(&*v);
          batch y_vec = batch::load_unaligned(&*w);
          y_vec       = a_vec * x_vec + y_vec;
          y_vec.store_unaligned(&*w);
        }
        p_loop_action();
      }
      p_log << "xsimd \t" << views::take(W, 3 * simd_width) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransformAligned(Index N = default_N) {
      // Do not change
      using batch               = xsimd::batch<Real>;
      constexpr auto simd_width = batch::size;
      Index          Nout       = min(N, default_Nout);
      // TODO
      std::array<Real, 3 * simd_width> V;
      std::array<Real, 3 * simd_width> W;

      Index remV   = reinterpret_cast<long long>(&V) % 64;
      Index remW   = reinterpret_cast<long long>(&W) % 64;
      Index startV = remV < 32 ? 32 - remV : 64 - remV;
      Index startW = remW < 32 ? 32 - remW : 64 - remW;
      startW       = (startW + sizeof(Real) - 1) / sizeof(Real);
      startV       = (startV + sizeof(Real) - 1) / sizeof(Real);
      // p_log << xsimd::is_aligned(&*(V.begin() + startV)) << '\n';
      // p_log << xsimd::is_aligned(&*(W.begin() + startW)) << '\n';
      // p_log << touches_single_cache_line(&*(W.begin() + startW), 32) << '\n';
      // p_log << touches_single_cache_line(&*(V.begin() + startV), 32) << '\n';
      std::iota(V.begin() + startV, V.end(), Real{1});
      std::iota(W.begin() + startW, W.end(), Real{1});
      for (auto _ : p_loop_state) {
        // using xsimd with aligned container
        batch a_vec(-1.0f);
        // TODO
        auto v = V.begin() + startV;
        auto w = W.begin() + startW;
        for (Index i = 0; i < N / simd_width; i++) {
          batch v_vec = batch::load_aligned(&*v);
          batch w_vec = batch::load_aligned(&*w);
          w_vec       = a_vec * v_vec + w_vec;
          w_vec.store_aligned(&*w);
        }
        p_loop_action();
      }
      p_log << "xsimd_aligned \t" << views::take(W, 3 * simd_width) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIterator(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
#pragma simd
        for (auto v = V.begin(); v != V.end(); ++v) {
          *w = a * (*v) + *w;
          ++w;
        }
        p_loop_action();
      }
      p_log << "OmpIterator \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIteratorInnerLoop(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      constexpr Index simd_width = 8;
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO
        auto w = Wview.begin();
        for (auto v = V.begin(); v != V.end();
             v += simd_width, w += simd_width) {
#pragma simd
          for (Index i = 0; i < simd_width; ++i) {
            *(w + i) = a * (*(v + i)) + *(w + i);
          }
        }
        p_loop_action();
      }
      p_log << "OmpIteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIteratorInnerLoop_index(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      constexpr Index simd_width = 8;
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});
      auto Wview = views::iota(0) | views::take(N) |
                   views::transform(
                     [&](auto i) -> Real & { return W[i % container_size]; });
      for (auto _ : p_loop_state) {
        // TODO

        auto w = Wview.begin();
        auto v = V.begin();
        for (auto j = 0; j < N; j += simd_width) {
#pragma simd
          for (Index i = 0; i < simd_width; ++i) {
            *(w + i + j) = a * (*(v + i + j)) + *(w + i + j);
          }
        }
        p_loop_action();
      }
      p_log << "OmpIteratorInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRange(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      for (auto _ : p_loop_state) {
// TODO
#pragma simd
        for (auto i : ranges::iota_view(0, N)) {
          W[i % container_size] = a * V[i] + W[i % container_size];
        }
        p_loop_action();
      }
      p_log << "OmpRange \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRange_index(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      for (auto _ : p_loop_state) {
// TODO
#pragma simd
        for (int i = 0; i < N; i++) {
          W[i % container_size] = a * V[i] + W[i % container_size];
        }
        p_loop_action();
      }
      p_log << "OmpRange \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRangeInnerLoop(Index N = default_N) {
      // Do not change
      Real  a    = -1.0f;
      Index Nout = min(N, default_Nout);
      // TODO
      constexpr Index simd_width = 8;
      auto            V = views::iota(1, 1 + N) |
                          views::transform([](auto i) -> Real { return i; });
      constexpr Index container_size = 256;
      Container<Real> W(container_size);
      std::iota(W.begin(), W.end(), Real{2});

      auto in = ranges::transform_view(
        ranges::iota_view(0, N / simd_width),
        [simd_width](Index s) { return s * simd_width; });
      for (auto _ : p_loop_state) {
        // TODO
        for (auto u : in) {
#pragma simd
          for (Index i = 0; i < simd_width; ++i) {
            W[(u + i) % container_size] =
              a * V[u + i] + W[(u + i) % container_size];
          }
        }
        p_loop_action();
      }
      // Do not change
      p_log << "OmpRangeInnerLoop \t" << views::take(W, Nout) << '\n';
      return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};

template<typename R, typename L>
array<char, 1> SIMD_transform_view<R, L>::p_one = {0};