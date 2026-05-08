#include "utilities.hpp"
#include <random>
#include <iostream>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <vector>
#include <deque>
#include <complex>
#include <algorithm>
#include <functional>
#include <ranges>
#include <execution>
#include <omp.h>

#include <xsimd/xsimd.hpp>

#pragma once

using namespace std;

template <typename R = array<char, 1>, typename L = function<void(void)>>
class SIMD_transform
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    SIMD_transform(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(2); }
    string get_log() { return p_log.str(); }

    using Index = int;
    using Int = int32_t;
    using Real = float;
    using CReal = complex<Real>;
    template <typename T>
    using Container = vector<T, allocator<T>>;
    template <typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec = execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n = 3;
    static constexpr Index default_m = 2;
    static constexpr Index default_N = 24;
    static constexpr Index default_Nout = 10;

    auto benchTransformIterator(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});

        Index Nout = min(N, default_Nout);
        // Main benchmarking loop
        for (auto _ : p_loop_state)
        {
            // Perform the transformation: W[i] = a * V[i] + W[i]
            for (auto v = V.begin(), w = W.begin(); v != V.end(); ++v, ++w)
            {
                *w = a * (*v) + *w;
            }
            p_loop_action(); // Benchmarking framework hook
        }
        // Log a subset of the result
        p_log << "Iterator \t" << views::take(W, Nout) << '\n';
        // Return memory usage: read (V and W) and write (W)
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        constexpr Index simd_width = 8;
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        Index Nout = min(N, default_Nout);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});

        for (auto _ : p_loop_state)
        {
            // Step through arrays in simd_width chunks — pattern mimics SIMD batch operations
            for (auto v = V.begin(), w = W.begin(); v != V.end();
                 v += simd_width, w += simd_width)
            {
                  // Manually unrolled loop body — emulates SIMD parallel execution
                for (Index i = 0; i < simd_width; ++i)
                {
                    *(w + i) = a * (*(v + i)) + *(w + i);
                }
            }
            p_loop_action();
        }
        p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRange(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // replace iterator with index
            for (auto i : ranges::iota_view(0, N))
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "Range \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        constexpr Index simd_width = 8;
        Index Nout = min(N, default_Nout);
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        auto in = ranges::transform_view(
            ranges::iota_view(0, N / simd_width),
            [simd_width](Index s)
            { return s * simd_width; });
        for (auto _ : p_loop_state)
        {
            // replace contianer with view
            for (auto u : in)
            {
                for (Index i = 0; i < simd_width; ++i)
                {
                    W[u + i] = a * V[u + i] + W[u + i];
                }
            }
            p_loop_action();
        }
        p_log << "RangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchTransformStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // use trnasform and lambda instead of loop
            std::transform(V.begin(), V.end(), W.begin(), W.begin(),
                           [a](const auto v, const auto w)
                           { return a * v + w; });
            p_loop_action();
        }
        p_log << "Stl \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformSimdStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);

        for (auto _ : p_loop_state)
        {
            // use trnasform with simd execution policy std::execution::unseq
            std::transform(std::execution::unseq, V.begin(), V.end(), W.begin(),
                           W.begin(),
                           [a](const auto v, const auto w)
                           { return a * v + w; });
            p_loop_action();
        }
        p_log << "SimdStl \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransform(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);

        for (auto _ : p_loop_state)
        {
            // using xsimd with unaligned container
            batch a_vec(-1.0f);
            for (auto v = V.begin(), w = W.begin(); v != V.end();
                 v += simd_width, w += simd_width)
            {
                batch x_vec = batch::load_unaligned(&*v);
                batch y_vec = batch::load_unaligned(&*w);
                y_vec = a_vec * x_vec + y_vec;
                y_vec.store_unaligned(&*w);
            }
            p_loop_action();
        }
        p_log << "xsimd \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransformAligned(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        AlignedContainer<Real> V(N);
        AlignedContainer<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);

        for (auto _ : p_loop_state)
        {
            // using xsimd with aligned container
            batch a_vec(-1.0f);
            for (auto v = V.begin(), w = W.begin(); v != V.end();
                 v += simd_width, w += simd_width)
            {
                batch v_vec = batch::load_aligned(&*v);
                batch w_vec = batch::load_aligned(&*w);
                w_vec = a_vec * v_vec + w_vec;
                w_vec.store_aligned(&*w);
            }
            p_loop_action();
        }
        p_log << "xsimd_aligned \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIterator(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp simd directive

            auto w = W.begin();
#pragma omp simd
            for (auto v = V.begin(); v != V.end(); ++v)
            {
                *w = a * (*v) + *w;
                ++w;
            }
            p_loop_action();
        }
        p_log << "OmpIterator \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchOmpSimdTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        constexpr Index simd_width = 8;
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp simd directive with manual grouping
            for (auto v = V.begin(), w = W.begin(); v != V.end();
                 v += simd_width, w += simd_width)
            {
#pragma omp simd
                for (Index i = 0; i < simd_width; ++i)
                {
                    *(w + i) = a * (*(v + i)) + *(w + i);
                }
            }
            p_loop_action();
        }
        p_log << "OmpIteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRange(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp simd directive with view index
#pragma omp simd
            for (auto i : ranges::iota_view(0, N))
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "OmpRange \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        constexpr Index simd_width = 8;
        Real a = -1.0f;
        Container<Real> V(N);
        Container<Real> W(N);
        std::iota(V.begin(), V.end(), Real{1});
        std::iota(W.begin(), W.end(), Real{2});
        Index Nout = min(N, default_Nout);
        auto in = ranges::transform_view(
            ranges::iota_view(0, N / simd_width),
            [simd_width](Index s)
            { return s * simd_width; });
        for (auto _ : p_loop_state)
        {
            // using openmp simd directive with nested loop
            for (auto u : in)
            {
#pragma omp simd
                for (Index i = 0; i < simd_width; ++i)
                {
                    W[u + i] = a * V[u + i] + W[u + i];
                }
            }
            p_loop_action();
        }
        p_log << "OmpRangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> SIMD_transform<R, L>::p_one = {0};