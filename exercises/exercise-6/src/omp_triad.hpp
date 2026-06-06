#include "allocator_adaptor.hpp"
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
#include "numa_extensions.hpp"

namespace NUMA_omp
{
    template <typename T>
    static void writePPM(T &buf, int width, int height, const char *fn)
    {
        FILE *fp = fopen(fn, "wb");
        fprintf(fp, "P6\n");
        fprintf(fp, "%d %d\n", width, height);
        fprintf(fp, "255\n");
        for (int i = 0; i < width * height; ++i)
        {
            // Map the iteration count to colors by just alternating between
            // two greys.
            char c = (buf[i] & 0x1) ? 240 : 20;
            for (int j = 0; j < 3; ++j)
                fputc(c, fp);
        }
        fclose(fp);
    }
    using namespace std;

    template <typename R = array<char, 1>, typename L = function<void(void)>>
    class Triad
    {
    private:
        static array<char, 1> p_one;
        stringstream p_log;
        R &p_loop_state;
        L p_loop_action;

    public:
        Triad(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << fixed << setprecision(2); }
        string get_log() { return p_log.str(); }
        template <typename T>
        struct triad_functor
        {
            const T a;
            triad_functor(T _a) : a{_a} {}
            T operator()(const T v, const T w) const { return a * v + w; }
        };
        using Index = int;
        using Int = int32_t;
        using Real = float;
        using LongReal = long double;
        using CReal = complex<Real>;
        using LongCReal = complex<LongReal>;
        template <typename T>
        using Container = vector<T, numa::no_init_allocator<T>>; // TODO use no-init-allocator this will cause old code crash
        // using Container = deque<T, allocator<T>>; // TODO use no-init-allocator
        template <typename T, int Fold = 1>
        using NUMAContainer = numa_extensions<Container<T>, Fold>;
        static constexpr auto stExec = execution::unseq;     // single-threaded execution policy
        static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy

        static constexpr Index default_n = 3;
        static constexpr Index default_m = 2;
        static constexpr Index default_N = 20;
        static constexpr Index default_Nout = 10;

        auto transform_ops_value(Index N = default_N)
        { // transform: individual operations with functors, mapping values to values - DONT
            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;

            NUMAContainer<Real> tmp(N, 1); // temporaries and many operations - DONT

            for (auto _ : p_loop_state)
            {
                // tmp = a;

#pragma omp parallel default(none) shared(tmp, W, a, V)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < tmp.size(); i++)
                    {
                    }
                // tmp = a * V;

#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < tmp.size(); i++)
                    {
                    }

// W = tmp + W;

#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_functor_value(Index N = default_N)
        { // transform: fused operations in one functor, mapping values to values

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;
            auto local_functor = triad_functor(a);
            for (auto _ : p_loop_state)
            {
// Z = a * V + W;
#pragma omp parallel default(none) shared(V, W, local_functor)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_value(Index N = default_N)
        { // transform: fused operations in one lambda, mapping values to values
            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
// Z = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_index(Index N = default_N)
        { // transform: fused operations in one lambda, mapping indices to values
            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
// Z = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_bad_init(Index N = default_N)
        {                                       // for_each fused operations with one lambda - DONT
            vector<Real> V(itSeq(0), itSeq(N)); // all data in one NUMA region - DONT
            vector<Real> W(N, 3);               // all data in one NUMA region - DONT
            Real a = 2;

            for (auto _ : p_loop_state)
            {
// Z = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_no_abstraction(Index N = default_N)
        { // for_each fused operations with one lambda - DONT
            Container<Real> V(N);
            std::copy(mtExec, itSeq(0), itSeq(N), begin(V)); // need tbb for intel compiler
            Container<Real> W(N, 3);
            std::fill(mtExec, begin(W), end(W), 3); // need tbb for intel compiler
            Real a = 2;

            for (auto _ : p_loop_state)
            {
// Z = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(Z), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << '\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }
    };
    template <typename R, typename L>
    array<char, 1> Triad<R, L>::p_one = {0};
}
