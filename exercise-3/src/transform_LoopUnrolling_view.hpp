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

#include "UNROLLFACTOR.h"

#pragma once

using namespace std;

template <typename R = array<char, 1>, typename L = function<void(void)>>
class transform_LoopUnrolling_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    transform_LoopUnrolling_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(3); }
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
    static constexpr Index default_Nout = 20;

    auto benchTransformOmpSimd(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;

        // TODO

        // --------

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp simd directive with index
#pragma omp simd
            for (Index i = 0; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "OmpIndex \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnroll(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;

        // TODO

        // --------

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp unroll directive with index
#pragma omp simd
#pragma unroll
            for (Index i = 0; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "DirectiveUnroll \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformDirectiveUnrollFactor64(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;

        // TODO

        // --------

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // using openmp unroll directive with index
#pragma omp simd
#pragma unroll(unroll_factor)
            for (Index i = 0; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "DirectiveUnrollFactor64 \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollManual(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;

        // TODO

        // --------

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            // Do not change
            // using manual unroll with index
#pragma omp simd
            for (Index i = 0; i < N; i += unroll_factor)
            {
                if (i + 0 < N)
                    W[i + 0] = a * V[i + 0] + W[i + 0];
                if (i + 1 < N)
                    W[i + 1] = a * V[i + 1] + W[i + 1];
                if (i + 2 < N)
                    W[i + 2] = a * V[i + 2] + W[i + 2];
                if (i + 3 < N)
                    W[i + 3] = a * V[i + 3] + W[i + 3];
                if (i + 4 < N)
                    W[i + 4] = a * V[i + 4] + W[i + 4];
                if (i + 5 < N)
                    W[i + 5] = a * V[i + 5] + W[i + 5];
                if (i + 6 < N)
                    W[i + 6] = a * V[i + 6] + W[i + 6];
                if (i + 7 < N)
                    W[i + 7] = a * V[i + 7] + W[i + 7];
                if (i + 8 < N)
                    W[i + 8] = a * V[i + 8] + W[i + 8];
                if (i + 9 < N)
                    W[i + 9] = a * V[i + 9] + W[i + 9];
                if (i + 10 < N)
                    W[i + 10] = a * V[i + 10] + W[i + 10];
                if (i + 11 < N)
                    W[i + 11] = a * V[i + 11] + W[i + 11];
                if (i + 12 < N)
                    W[i + 12] = a * V[i + 12] + W[i + 12];
                if (i + 13 < N)
                    W[i + 13] = a * V[i + 13] + W[i + 13];
                if (i + 14 < N)
                    W[i + 14] = a * V[i + 14] + W[i + 14];
                if (i + 15 < N)
                    W[i + 15] = a * V[i + 15] + W[i + 15];
                if (i + 16 < N)
                    W[i + 16] = a * V[i + 16] + W[i + 16];
                if (i + 17 < N)
                    W[i + 17] = a * V[i + 17] + W[i + 17];
                if (i + 18 < N)
                    W[i + 18] = a * V[i + 18] + W[i + 18];
                if (i + 19 < N)
                    W[i + 19] = a * V[i + 19] + W[i + 19];
                if (i + 20 < N)
                    W[i + 20] = a * V[i + 20] + W[i + 20];
                if (i + 21 < N)
                    W[i + 21] = a * V[i + 21] + W[i + 21];
                if (i + 22 < N)
                    W[i + 22] = a * V[i + 22] + W[i + 22];
                if (i + 23 < N)
                    W[i + 23] = a * V[i + 23] + W[i + 23];
                if (i + 24 < N)
                    W[i + 24] = a * V[i + 24] + W[i + 24];
                if (i + 25 < N)
                    W[i + 25] = a * V[i + 25] + W[i + 25];
                if (i + 26 < N)
                    W[i + 26] = a * V[i + 26] + W[i + 26];
                if (i + 27 < N)
                    W[i + 27] = a * V[i + 27] + W[i + 27];
                if (i + 28 < N)
                    W[i + 28] = a * V[i + 28] + W[i + 28];
                if (i + 29 < N)
                    W[i + 29] = a * V[i + 29] + W[i + 29];
                if (i + 30 < N)
                    W[i + 30] = a * V[i + 30] + W[i + 30];
                if (i + 31 < N)
                    W[i + 31] = a * V[i + 31] + W[i + 31];
                if (i + 32 < N)
                    W[i + 32] = a * V[i + 32] + W[i + 32];
                if (i + 33 < N)
                    W[i + 33] = a * V[i + 33] + W[i + 33];
                if (i + 34 < N)
                    W[i + 34] = a * V[i + 34] + W[i + 34];
                if (i + 35 < N)
                    W[i + 35] = a * V[i + 35] + W[i + 35];
                if (i + 36 < N)
                    W[i + 36] = a * V[i + 36] + W[i + 36];
                if (i + 37 < N)
                    W[i + 37] = a * V[i + 37] + W[i + 37];
                if (i + 38 < N)
                    W[i + 38] = a * V[i + 38] + W[i + 38];
                if (i + 39 < N)
                    W[i + 39] = a * V[i + 39] + W[i + 39];
                if (i + 40 < N)
                    W[i + 40] = a * V[i + 40] + W[i + 40];
                if (i + 41 < N)
                    W[i + 41] = a * V[i + 41] + W[i + 41];
                if (i + 42 < N)
                    W[i + 42] = a * V[i + 42] + W[i + 42];
                if (i + 43 < N)
                    W[i + 43] = a * V[i + 43] + W[i + 43];
                if (i + 44 < N)
                    W[i + 44] = a * V[i + 44] + W[i + 44];
                if (i + 45 < N)
                    W[i + 45] = a * V[i + 45] + W[i + 45];
                if (i + 46 < N)
                    W[i + 46] = a * V[i + 46] + W[i + 46];
                if (i + 47 < N)
                    W[i + 47] = a * V[i + 47] + W[i + 47];
                if (i + 48 < N)
                    W[i + 48] = a * V[i + 48] + W[i + 48];
                if (i + 49 < N)
                    W[i + 49] = a * V[i + 49] + W[i + 49];
                if (i + 50 < N)
                    W[i + 50] = a * V[i + 50] + W[i + 50];
                if (i + 51 < N)
                    W[i + 51] = a * V[i + 51] + W[i + 51];
                if (i + 52 < N)
                    W[i + 52] = a * V[i + 52] + W[i + 52];
                if (i + 53 < N)
                    W[i + 53] = a * V[i + 53] + W[i + 53];
                if (i + 54 < N)
                    W[i + 54] = a * V[i + 54] + W[i + 54];
                if (i + 55 < N)
                    W[i + 55] = a * V[i + 55] + W[i + 55];
                if (i + 56 < N)
                    W[i + 56] = a * V[i + 56] + W[i + 56];
                if (i + 57 < N)
                    W[i + 57] = a * V[i + 57] + W[i + 57];
                if (i + 58 < N)
                    W[i + 58] = a * V[i + 58] + W[i + 58];
                if (i + 59 < N)
                    W[i + 59] = a * V[i + 59] + W[i + 59];
                if (i + 60 < N)
                    W[i + 60] = a * V[i + 60] + W[i + 60];
                if (i + 61 < N)
                    W[i + 61] = a * V[i + 61] + W[i + 61];
                if (i + 62 < N)
                    W[i + 62] = a * V[i + 62] + W[i + 62];
                if (i + 63 < N)
                    W[i + 63] = a * V[i + 63] + W[i + 63];
            }
            p_loop_action();
        }
        p_log << "UnrollManual \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollLoopPeeling(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;

        // TODO

        // --------
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                W[i + 0] = a * V[i + 0] + W[i + 0];
                W[i + 1] = a * V[i + 1] + W[i + 1];
                W[i + 2] = a * V[i + 2] + W[i + 2];
                W[i + 3] = a * V[i + 3] + W[i + 3];
                W[i + 4] = a * V[i + 4] + W[i + 4];
                W[i + 5] = a * V[i + 5] + W[i + 5];
                W[i + 6] = a * V[i + 6] + W[i + 6];
                W[i + 7] = a * V[i + 7] + W[i + 7];
                W[i + 8] = a * V[i + 8] + W[i + 8];
                W[i + 9] = a * V[i + 9] + W[i + 9];
                W[i + 10] = a * V[i + 10] + W[i + 10];
                W[i + 11] = a * V[i + 11] + W[i + 11];
                W[i + 12] = a * V[i + 12] + W[i + 12];
                W[i + 13] = a * V[i + 13] + W[i + 13];
                W[i + 14] = a * V[i + 14] + W[i + 14];
                W[i + 15] = a * V[i + 15] + W[i + 15];
                W[i + 16] = a * V[i + 16] + W[i + 16];
                W[i + 17] = a * V[i + 17] + W[i + 17];
                W[i + 18] = a * V[i + 18] + W[i + 18];
                W[i + 19] = a * V[i + 19] + W[i + 19];
                W[i + 20] = a * V[i + 20] + W[i + 20];
                W[i + 21] = a * V[i + 21] + W[i + 21];
                W[i + 22] = a * V[i + 22] + W[i + 22];
                W[i + 23] = a * V[i + 23] + W[i + 23];
                W[i + 24] = a * V[i + 24] + W[i + 24];
                W[i + 25] = a * V[i + 25] + W[i + 25];
                W[i + 26] = a * V[i + 26] + W[i + 26];
                W[i + 27] = a * V[i + 27] + W[i + 27];
                W[i + 28] = a * V[i + 28] + W[i + 28];
                W[i + 29] = a * V[i + 29] + W[i + 29];
                W[i + 30] = a * V[i + 30] + W[i + 30];
                W[i + 31] = a * V[i + 31] + W[i + 31];
                W[i + 32] = a * V[i + 32] + W[i + 32];
                W[i + 33] = a * V[i + 33] + W[i + 33];
                W[i + 34] = a * V[i + 34] + W[i + 34];
                W[i + 35] = a * V[i + 35] + W[i + 35];
                W[i + 36] = a * V[i + 36] + W[i + 36];
                W[i + 37] = a * V[i + 37] + W[i + 37];
                W[i + 38] = a * V[i + 38] + W[i + 38];
                W[i + 39] = a * V[i + 39] + W[i + 39];
                W[i + 40] = a * V[i + 40] + W[i + 40];
                W[i + 41] = a * V[i + 41] + W[i + 41];
                W[i + 42] = a * V[i + 42] + W[i + 42];
                W[i + 43] = a * V[i + 43] + W[i + 43];
                W[i + 44] = a * V[i + 44] + W[i + 44];
                W[i + 45] = a * V[i + 45] + W[i + 45];
                W[i + 46] = a * V[i + 46] + W[i + 46];
                W[i + 47] = a * V[i + 47] + W[i + 47];
                W[i + 48] = a * V[i + 48] + W[i + 48];
                W[i + 49] = a * V[i + 49] + W[i + 49];
                W[i + 50] = a * V[i + 50] + W[i + 50];
                W[i + 51] = a * V[i + 51] + W[i + 51];
                W[i + 52] = a * V[i + 52] + W[i + 52];
                W[i + 53] = a * V[i + 53] + W[i + 53];
                W[i + 54] = a * V[i + 54] + W[i + 54];
                W[i + 55] = a * V[i + 55] + W[i + 55];
                W[i + 56] = a * V[i + 56] + W[i + 56];
                W[i + 57] = a * V[i + 57] + W[i + 57];
                W[i + 58] = a * V[i + 58] + W[i + 58];
                W[i + 59] = a * V[i + 59] + W[i + 59];
                W[i + 60] = a * V[i + 60] + W[i + 60];
                W[i + 61] = a * V[i + 61] + W[i + 61];
                W[i + 62] = a * V[i + 62] + W[i + 62];
                W[i + 63] = a * V[i + 63] + W[i + 63];
            }
#pragma omp simd
            for (Index i = N - rem; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "UnrollLoopPeeling\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformUnrollLoopPeelingDirective(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;

        constexpr auto unroll_factor = UNROLLFACTOR;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;

        // TODO

        // --------

        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            batch a_vec(-1.0f);

#pragma omp simd
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {

#pragma unroll
                
                // TODO

                // --------
            }
#pragma omp simd
            for (Index i = N - rem; i < N; i++)
            {
                W[i] = a * V[i] + W[i];
            }
            p_loop_action();
        }
        p_log << "UnrollLoopPeelingDirective\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> transform_LoopUnrolling_view<R, L>::p_one = {0};
