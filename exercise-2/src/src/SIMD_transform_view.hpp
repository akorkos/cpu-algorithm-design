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
class SIMD_transform_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    SIMD_transform_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(2); }
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
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }

        p_log << "Iterator \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }

        p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRange(Index N = default_N)
    {

        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }

        p_log << "Range \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        constexpr Index simd_width = 8;
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }

        p_log << "RangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchTransformStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }

        p_log << "Stl \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformSimdStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

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
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // using xsimd with aligned container
            batch a_vec(-1.0f);
            // TODO

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
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // using xsimd with aligned container
            batch a_vec(-1.0f);
            // TODO

            p_loop_action();
        }
        p_log << "xsimd_aligned \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIterator(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }
        p_log << "OmpIterator \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchOmpSimdTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }
        p_log << "OmpIteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRange(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }
        p_log << "OmpRange \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        // TODO

        for (auto _ : p_loop_state)
        {
            // TODO

            p_loop_action();
        }
        // Do not change
        p_log << "OmpRangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> SIMD_transform_view<R, L>::p_one = {0};