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
class reduce_LoopUnrolling_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    reduce_LoopUnrolling_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(3); }
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

    auto benchReduceOmpSimd(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;

        // TODO

        // --------
        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // reduce with openmp directive
#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N; i++)
            {
                sum += V[i];
            }
        }
        p_log << "OmpIndex \t" << sum << '\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceDirectiveUnroll(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        // TODO

        // --------
        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using openmp unroll directive with index
#pragma omp simd reduction(+ : sum)
#pragma unroll
            for (Index i = 0; i < N; i++)
            {
                sum += V[i];
            }
        }
        p_log << "DirectiveUnroll \t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceDirectiveUnrollFactor64(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using openmp unroll directive with pre-set factor and index

#pragma omp simd reduction(+ : sum)
#pragma unroll(unroll_factor)
            for (Index i = 0; i < N; i++)
            {
                sum += V[i];
            }
        }
        p_log << "DirectiveUnrollFactor64 \t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceUnrollManual(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using manual unroll with index

#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N; i += unroll_factor)
            {
                if (i + 0 < N)
                    sum += V[i + 0];
                if (i + 1 < N)
                    sum += V[i + 1];
                if (i + 2 < N)
                    sum += V[i + 2];
                if (i + 3 < N)
                    sum += V[i + 3];
                if (i + 4 < N)
                    sum += V[i + 4];
                if (i + 5 < N)
                    sum += V[i + 5];
                if (i + 6 < N)
                    sum += V[i + 6];
                if (i + 7 < N)
                    sum += V[i + 7];
                if (i + 8 < N)
                    sum += V[i + 8];
                if (i + 9 < N)
                    sum += V[i + 9];
                if (i + 10 < N)
                    sum += V[i + 10];
                if (i + 11 < N)
                    sum += V[i + 11];
                if (i + 12 < N)
                    sum += V[i + 12];
                if (i + 13 < N)
                    sum += V[i + 13];
                if (i + 14 < N)
                    sum += V[i + 14];
                if (i + 15 < N)
                    sum += V[i + 15];
                if (i + 16 < N)
                    sum += V[i + 16];
                if (i + 17 < N)
                    sum += V[i + 17];
                if (i + 18 < N)
                    sum += V[i + 18];
                if (i + 19 < N)
                    sum += V[i + 19];
                if (i + 20 < N)
                    sum += V[i + 20];
                if (i + 21 < N)
                    sum += V[i + 21];
                if (i + 22 < N)
                    sum += V[i + 22];
                if (i + 23 < N)
                    sum += V[i + 23];
                if (i + 24 < N)
                    sum += V[i + 24];
                if (i + 25 < N)
                    sum += V[i + 25];
                if (i + 26 < N)
                    sum += V[i + 26];
                if (i + 27 < N)
                    sum += V[i + 27];
                if (i + 28 < N)
                    sum += V[i + 28];
                if (i + 29 < N)
                    sum += V[i + 29];
                if (i + 30 < N)
                    sum += V[i + 30];
                if (i + 31 < N)
                    sum += V[i + 31];
                if (i + 32 < N)
                    sum += V[i + 32];
                if (i + 33 < N)
                    sum += V[i + 33];
                if (i + 34 < N)
                    sum += V[i + 34];
                if (i + 35 < N)
                    sum += V[i + 35];
                if (i + 36 < N)
                    sum += V[i + 36];
                if (i + 37 < N)
                    sum += V[i + 37];
                if (i + 38 < N)
                    sum += V[i + 38];
                if (i + 39 < N)
                    sum += V[i + 39];
                if (i + 40 < N)
                    sum += V[i + 40];
                if (i + 41 < N)
                    sum += V[i + 41];
                if (i + 42 < N)
                    sum += V[i + 42];
                if (i + 43 < N)
                    sum += V[i + 43];
                if (i + 44 < N)
                    sum += V[i + 44];
                if (i + 45 < N)
                    sum += V[i + 45];
                if (i + 46 < N)
                    sum += V[i + 46];
                if (i + 47 < N)
                    sum += V[i + 47];
                if (i + 48 < N)
                    sum += V[i + 48];
                if (i + 49 < N)
                    sum += V[i + 49];
                if (i + 50 < N)
                    sum += V[i + 50];
                if (i + 51 < N)
                    sum += V[i + 51];
                if (i + 52 < N)
                    sum += V[i + 52];
                if (i + 53 < N)
                    sum += V[i + 53];
                if (i + 54 < N)
                    sum += V[i + 54];
                if (i + 55 < N)
                    sum += V[i + 55];
                if (i + 56 < N)
                    sum += V[i + 56];
                if (i + 57 < N)
                    sum += V[i + 57];
                if (i + 58 < N)
                    sum += V[i + 58];
                if (i + 59 < N)
                    sum += V[i + 59];
                if (i + 60 < N)
                    sum += V[i + 60];
                if (i + 61 < N)
                    sum += V[i + 61];
                if (i + 62 < N)
                    sum += V[i + 62];
                if (i + 63 < N)
                    sum += V[i + 63];
            }
            p_loop_action();
        }
        p_log << "UnrollManual \t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceUnrollLoopPeeling(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using manual unroll with index and follow up loop

#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                sum += V[i + 0];
                sum += V[i + 1];
                sum += V[i + 2];
                sum += V[i + 3];
                sum += V[i + 4];
                sum += V[i + 5];
                sum += V[i + 6];
                sum += V[i + 7];
                sum += V[i + 8];
                sum += V[i + 9];
                sum += V[i + 10];
                sum += V[i + 11];
                sum += V[i + 12];
                sum += V[i + 13];
                sum += V[i + 14];
                sum += V[i + 15];
                sum += V[i + 16];
                sum += V[i + 17];
                sum += V[i + 18];
                sum += V[i + 19];
                sum += V[i + 20];
                sum += V[i + 21];
                sum += V[i + 22];
                sum += V[i + 23];
                sum += V[i + 24];
                sum += V[i + 25];
                sum += V[i + 26];
                sum += V[i + 27];
                sum += V[i + 28];
                sum += V[i + 29];
                sum += V[i + 30];
                sum += V[i + 31];
                sum += V[i + 32];
                sum += V[i + 33];
                sum += V[i + 34];
                sum += V[i + 35];
                sum += V[i + 36];
                sum += V[i + 37];
                sum += V[i + 38];
                sum += V[i + 39];
                sum += V[i + 40];
                sum += V[i + 41];
                sum += V[i + 42];
                sum += V[i + 43];
                sum += V[i + 44];
                sum += V[i + 45];
                sum += V[i + 46];
                sum += V[i + 47];
                sum += V[i + 48];
                sum += V[i + 49];
                sum += V[i + 50];
                sum += V[i + 51];
                sum += V[i + 52];
                sum += V[i + 53];
                sum += V[i + 54];
                sum += V[i + 55];
                sum += V[i + 56];
                sum += V[i + 57];
                sum += V[i + 58];
                sum += V[i + 59];
                sum += V[i + 60];
                sum += V[i + 61];
                sum += V[i + 62];
                sum += V[i + 63];
            }
#pragma omp simd reduction(+ : sum)
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
            p_loop_action();
        }
        p_log << "UnrollLoopPeeling\t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceUnrollLoopPeelingDirective(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------
        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using manual unroll directive with index and follow up loop

#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
#pragma unroll
                for (Index j = 0; j < unroll_factor; j++)
                {
                    sum += V[i + j];
                }
            }
#pragma omp simd reduction(+ : sum)
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
            p_loop_action();
        }
        p_log << "UnrollLoopPeelingDirective\t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceUnrollTree(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // using manual unroll directive with index and follow up loop
            array<Real, 16> local_sum = {0};
#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                // level 1
                local_sum[0] = V[i + 0 * 4 + 0] + V[i + 0 * 4 + 1] + V[i + 0 * 4 + 2] + V[i + 0 * 4 + 3];
                local_sum[1] = V[i + 1 * 4 + 0] + V[i + 1 * 4 + 1] + V[i + 1 * 4 + 2] + V[i + 1 * 4 + 3];
                local_sum[2] = V[i + 2 * 4 + 0] + V[i + 2 * 4 + 1] + V[i + 2 * 4 + 2] + V[i + 2 * 4 + 3];
                local_sum[3] = V[i + 3 * 4 + 0] + V[i + 3 * 4 + 1] + V[i + 3 * 4 + 2] + V[i + 3 * 4 + 3];
                local_sum[4] = V[i + 4 * 4 + 0] + V[i + 4 * 4 + 1] + V[i + 4 * 4 + 2] + V[i + 4 * 4 + 3];
                local_sum[5] = V[i + 5 * 4 + 0] + V[i + 5 * 4 + 1] + V[i + 5 * 4 + 2] + V[i + 5 * 4 + 3];
                local_sum[6] = V[i + 6 * 4 + 0] + V[i + 6 * 4 + 1] + V[i + 6 * 4 + 2] + V[i + 6 * 4 + 3];
                local_sum[7] = V[i + 7 * 4 + 0] + V[i + 7 * 4 + 1] + V[i + 7 * 4 + 2] + V[i + 7 * 4 + 3];
                local_sum[8] = V[i + 8 * 4 + 0] + V[i + 8 * 4 + 1] + V[i + 8 * 4 + 2] + V[i + 8 * 4 + 3];
                local_sum[9] = V[i + 9 * 4 + 0] + V[i + 9 * 4 + 1] + V[i + 9 * 4 + 2] + V[i + 9 * 4 + 3];
                local_sum[10] = V[i + 10 * 4 + 0] + V[i + 10 * 4 + 1] + V[i + 10 * 4 + 2] + V[i + 10 * 4 + 3];
                local_sum[11] = V[i + 11 * 4 + 0] + V[i + 11 * 4 + 1] + V[i + 11 * 4 + 2] + V[i + 11 * 4 + 3];
                local_sum[12] = V[i + 12 * 4 + 0] + V[i + 12 * 4 + 1] + V[i + 12 * 4 + 2] + V[i + 12 * 4 + 3];
                local_sum[13] = V[i + 13 * 4 + 0] + V[i + 13 * 4 + 1] + V[i + 13 * 4 + 2] + V[i + 13 * 4 + 3];
                local_sum[14] = V[i + 14 * 4 + 0] + V[i + 14 * 4 + 1] + V[i + 14 * 4 + 2] + V[i + 14 * 4 + 3];
                local_sum[15] = V[i + 15 * 4 + 0] + V[i + 15 * 4 + 1] + V[i + 15 * 4 + 2] + V[i + 15 * 4 + 3];

                // level 2
                local_sum[0] = local_sum[0 * 4 + 0] + local_sum[0 * 4 + 1] + local_sum[0 * 4 + 2] + local_sum[0 * 4 + 3];
                local_sum[1] = local_sum[1 * 4 + 0] + local_sum[1 * 4 + 1] + local_sum[1 * 4 + 2] + local_sum[1 * 4 + 3];
                local_sum[2] = local_sum[2 * 4 + 0] + local_sum[2 * 4 + 1] + local_sum[2 * 4 + 2] + local_sum[2 * 4 + 3];
                local_sum[3] = local_sum[3 * 4 + 0] + local_sum[3 * 4 + 1] + local_sum[3 * 4 + 2] + local_sum[3 * 4 + 3];

                // level 3
                // local_sum[0] = local_sum[0] + local_sum[1] + local_sum[2] + local_sum[3];
                // sum += local_sum[0];

                sum += local_sum[0] + local_sum[1] + local_sum[2] + local_sum[3];
            }

#pragma omp simd reduction(+ : sum)
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
            p_loop_action();
        }
        p_log << "UnrollTree\t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchReduceUnrollTreeDirective(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        constexpr auto tree_deg = 8;
        // TODO

        // --------
        Real sum;
        Index Nout = min(N, default_Nout);
        for (auto _ : p_loop_state)
        {
            sum = 0;
            // TODO

            // --------
#pragma omp simd reduction(+ : sum)
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                // TODO

                // --------
            }

#pragma omp simd reduction(+ : sum)
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
            p_loop_action();
        }
        p_log << "UnrollTreeDirective\t" << sum << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchReduceUnrollSimdXHorizontal(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (Index i = 0; i < N - rem; i += unroll_factor)
            {
                // TODO

                // --------
            }
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
        }
        p_log << "Unrollxsimd_Horizontal \t" << sum << '\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceUnrollSimdXVertical(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        constexpr auto unroll_factor = 64;
        N = N % unroll_factor ? N : N + 1;
        auto rem = N % unroll_factor;
        // TODO

        // --------

        Real sum;
        Index Nout = min(N, default_Nout);
        batch sum_vec;

        for (auto _ : p_loop_state)
        {
            sum = 0;
            sum_vec = 0;
            // TODO

            for (Index i = 0; i < N - rem; i += unroll_factor)
            {


            }

            // --------
            for (Index i = N - rem; i < N; i++)
            {
                sum += V[i];
            }
        }
        p_log << "Unrollxsimd_Vertical \t" << sum << '\n';
        return tuple{N * sizeof(Real), 0};
    }
};
template <typename R, typename L>
array<char, 1> reduce_LoopUnrolling_view<R, L>::p_one = {0};
