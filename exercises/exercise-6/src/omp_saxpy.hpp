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
    class Saxpy
    {
    private:
        static array<char, 1> p_one;
        stringstream p_log;
        R &p_loop_state;
        L p_loop_action;

    public:
        Saxpy(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << fixed << setprecision(2); }
        string get_log() { return p_log.str(); }
        template <typename T>
        struct saxpy_functor
        {
            const T a;
            saxpy_functor(T _a) : a{_a} {}
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

#pragma omp parallel default(none) shared(tmp, a, V, W)
                {
#pragma omp for simd
#pragma unroll
                    for (int i = 0; i < tmp.size(); i++)
                    {
                        tmp[i] = a;
                    }
                    // }
                    // tmp = a * V;
                    // #pragma omp parallel default(none) shared(tmp, V)
                    // {
#pragma omp for simd
#pragma unroll
                    for (int i = 0; i < tmp.size(); i++)
                    {
                        tmp[i] = tmp[i] * V[i];
                    }
                    //                 // }
                    // // W = tmp + W;
                    // // #pragma omp parallel default(none) shared(tmp, W)
                    //                 // {
#pragma omp for simd
#pragma unroll
                    for (int i = 0; i < W.size(); i++)
                    {
                        W[i] = tmp[i] + W[i];
                    }
                }
                p_loop_action();
            }

            // output
            std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
            p_log << "\n" <<N<<"\n";
            // input and output size in bytes
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';

            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_functor_value(Index N = default_N)
        { // transform: fused operations in one functor, mapping values to values

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;
            auto local_functor = saxpy_functor(a);
            for (auto _ : p_loop_state)
            {
// W = a * V + W;
#pragma omp parallel default(none) shared(V, W, local_functor)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = local_functor(V[i], W[i]);
                    }
                }
                p_loop_action();
            }

            // output
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';
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
// W = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = [a](auto v, auto w)
                        { return a * v + w; }(V[i], W[i]);
                    }
                }
                p_loop_action();
            }

            // input and output size in bytes
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_index(Index N = default_N)
        { // transform: fused operations in one lambda, mapping indices to values
            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, 3);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
// W = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = a * V[i] + W[i];
                    }
                }
                p_loop_action();
            }

            // output
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }

        auto transform_lambda_no_abstraction(Index N = default_N)
        { // use container instead of numa container
            Container<Real> V(N);
            std::copy(mtExec, itSeq(0), itSeq(N), begin(V)); // need tbb for intel compiler
            Container<Real> W(N, 3);
            std::fill(mtExec, begin(W), end(W), 3); // need tbb for intel compiler
            Real a = 2;
            for (auto _ : p_loop_state)
            {
// W = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = a * V[i] + W[i];
                    }
                }
                p_loop_action();
            }

            // output
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';
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
// W = a * V + W;
#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = a * V[i] + W[i];
                    }
                }
                p_loop_action();
            }

            // output
            p_log<<"saxpy:\t"<<views::take(W,default_Nout)<<'\n';
            // input and output size in bytes
            return std::tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
        }
        auto transform_mandelbrot_fixed(Index N = default_N)
        {
            int height = std::sqrt(N >> 5);
            int width = height / 2 * 3;
            constexpr Real x0 = -2;
            constexpr Real y0 = -1;
            constexpr Real x1 = 1;
            constexpr Real y1 = 1;
            constexpr int maxIter = 16;
            Real dx = (x1 - x0) / width;
            Real dy = (y1 - y0) / height;

            NUMAContainer<int> resVec(height * width, 0);
            // long int totaliter =0;
            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(resVec, height, width, dx, dy)
                {
#pragma omp for
                    for (int j = 0; j < height; j++)
                    {
#pragma omp simd
                        for (int i = 0; i < width; ++i)
                        {
                            Real x = x0 + i * dx;
                            Real y = y0 + j * dy;
                            int index = (j * width + i);
                            CReal z{x, y};
                            int itercount;
#pragma unroll(16)
                            for (itercount = 0; itercount < maxIter; ++itercount)
                            {
                                // if (std::abs(z) > 4.)
                                //     break;
                                z = z * z + CReal{x, y};
                            }
                            resVec[index] = z.imag() * z.imag() + z.real() * z.real() > 4. ? itercount : 0;
                        }
                    }
                }
                p_loop_action();
            }

            writePPM(resVec, width, height, "mandelbrot.ppm");
            long int totaliter = std::accumulate(resVec.begin(), resVec.end(), static_cast<long int>(0));
            return std::tuple{static_cast<long long>((8 * maxIter + 10) * width) * height, 0};
        }
        auto transform_mandelbrot(Index N = default_N)
        {
            int height = std::sqrt(N >> 5);
            int width = height / 2 * 3;
            constexpr Real x0 = -2;
            constexpr Real y0 = -1;
            constexpr Real x1 = 1;
            constexpr Real y1 = 1;
            constexpr int maxIter = 1024;
            Real dx = (x1 - x0) / width;
            Real dy = (y1 - y0) / height;

            NUMAContainer<int> resVec(height * width, 0);
            // long int totaliter =0;
            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(resVec, height, width, dx, dy)
                {
#pragma omp for
                    for (int j = 0; j < height; j++)
                    {
#pragma omp simd
                        for (int i = 0; i < width; ++i)
                        {
                            Real x = x0 + i * dx;
                            Real y = y0 + j * dy;
                            int index = (j * width + i);
                            CReal z{x, y};
                            int itercount;
                            for (itercount = 0; itercount < maxIter; itercount += 16)
                            {
                                // int local_iter_count = 0;
#pragma unroll(16)
                                for (int k = 0; k < 8; k++)
                                {
                                    z = z * z + CReal{x, y};
                                    // local_iter_count += (std::abs(z) > 4.);
                                }

                                if (std::abs(z) > 4.)
                                    break;
                            }
                            resVec[index] = itercount;
                        }
                    }
                }
                p_loop_action();
            }

            writePPM(resVec, width, height, "mandelbrot.ppm");
            long int totaliter = std::accumulate(resVec.begin(), resVec.end(), static_cast<long int>(0));
            return std::tuple{totaliter*6 + (width*height)*5, 0};
        }
    };
    template <typename R, typename L>
    array<char, 1> Saxpy<R, L>::p_one = {0};
}
