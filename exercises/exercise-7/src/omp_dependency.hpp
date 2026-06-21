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

namespace NUMA_omp {
template<typename T>
static void writePPM(T& buf, int width, int height, char const* fn)
{
    FILE* fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (int j = 0; j < 3; ++j) {
            fputc(c, fp);
        }
    }
    fclose(fp);
}

#pragma omp declare simd

template<typename T>
T func(T x)
{
    return x * x;
}

#pragma omp declare simd

template<typename T>
T func2(T x)
{
    return x * x + 2;
}

using namespace std;

template<typename R = array<char, 1>, typename L = function<void(void)>>
class Dependency
{
private:
    static array<char, 1> p_one;
    stringstream p_log;
    R& p_loop_state;
    L p_loop_action;

public:
    Dependency(R& loop_state = p_one, L loop_action = []() { }) : p_loop_state(loop_state), p_loop_action(loop_action)
    {
        p_log << fixed << setprecision(2);
    }

    string get_log() { return p_log.str(); }

    template<typename T>
    struct triad_functor
    {
        T const a;

        triad_functor(T _a) : a {_a} { }

        T operator()(T const v, T const w) const { return a * v + w; }
    };

    using Index = int;
    using Int = int32_t;
    using Real = float;
    using LongReal = long double;
    using CReal = complex<Real>;
    using LongCReal = complex<LongReal>;
    template<typename T>
    using Container
        = vector<T, numa::no_init_allocator<T>>; // TODO use no-init-allocator this will cause old code crash
    // using Container = deque<T, allocator<T>>; // TODO use no-init-allocator
    template<typename T, int Fold = 1>
    using NUMAContainer = numa_extensions<Container<T>, Fold>;
    static constexpr auto stExec = execution::unseq; // single-threaded execution policy
    static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n = 3;
    static constexpr Index default_m = 2;
    static constexpr Index default_N = 20;
    static constexpr Index default_Nout = 10;

    auto update_sequential(Index N = default_N)
    {
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;
        constexpr Int stride = 1024;

        for (auto _ : p_loop_state) {
            // Z = a * V + W;
            {
                for (Index i = 0; i < V.size() - stride; i++) {
                    V[i + stride] = func(V[i]);
                }
            }
            p_loop_action();
        }

        // output
        std::copy_n(cbegin(V), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto update_parallel_simd(Index N = default_N)
    { 
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;
        constexpr int stride = 1024;

        for (auto _ : p_loop_state) {
            for (Index i = 0; i < V.size() - stride; i += stride) {

                #pragma omp simd
                for (Index j = 0; j < stride; ++j) {
                    V[i + j + stride] = func(V[i + j]);
                }
            }
                
            p_loop_action();
        }
        // output
        std::copy_n(cbegin(V), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto update_parallel_nested_loop(Index N = default_N)
    { 
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;
        constexpr int stride = 1024;

        for (auto _ : p_loop_state) {
            for (Index i = 0; i < V.size() - stride; i += stride) {

                #pragma omp parallel for simd
                for (Index j = 0; j < stride; ++j) {
                    V[i + j + stride] = func(V[i + j]);
                }
            }
                
            p_loop_action();
        }

        // output
        std::copy_n(cbegin(V), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto between_iter_sequential(Index N = default_N)
    { // transform: fused operations in one lambda, mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;

        for (auto _ : p_loop_state) {
            V[0] = 0;
            for (size_t i = 1; i < V.size(); i++) {
                V[i] = func(i);
                W[i] = func2(V[i - 1]);
            }
            p_loop_action();
        }

        // output
        std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto between_iter_twoloops(Index N = default_N)
    { // transform: fused operations in one lambda, mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;

        for (auto _ : p_loop_state) {
            V[0] = 0;
            W[0] = 0;

            #pragma omp parallel for 
            for (Index i = 1; i < V.size(); i++) {
                V[i] = func(i);
            }

            #pragma omp parallel for 
            for (Index i = 1; i < V.size(); i++) {
                W[i] = func2(V[i - 1]);
            }

            p_loop_action();
        }

        // output
        std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto between_iter_optimal(Index N = default_N)
    { // transform: fused operations in one lambda, mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);
        Real a = 2;

        for (auto _ : p_loop_state) {
            V[0] = 0;
            W[0] = 0;

            for (Index i = 1; i < V.size() - 1; i++) {
                V[i] = func(i);
                W[i + 1] = func2(V[i]);
            }

            Index last = V.size() - 1;

            V[last] = func(V[last]);

            p_loop_action();
        }

        // output
        std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }


    auto between_iter_manual_split(Index N = default_N)
    { // transform: fused operations in one lambda, mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);

        for (auto _ : p_loop_state) {
            V[0] = 0;
            W[0] = 0;
            const auto work_items = V.size() > 1 ? V.size() - 1 : 0;
#pragma omp parallel default(none) shared(V, W, work_items)
            {
                const auto thread_count = static_cast<size_t>(omp_get_num_threads());
                const auto thread_id = static_cast<size_t>(omp_get_thread_num());
                const auto first = size_t {1} + (work_items * thread_id) / thread_count;
                const auto last = size_t {1} + (work_items * (thread_id + 1)) / thread_count;

                if (first < last) {
                    auto previous_v = first == 1 ? Real {0} : static_cast<Real>(func(first - 1));
                    for (size_t i = first; i < last; i++) {
                        const auto current_v = static_cast<Real>(func(i));
                        V[i] = current_v;
                        W[i] = func2(previous_v);
                        previous_v = current_v;
                    }
                }
            }
            p_loop_action();
        }

        // output
        std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }

    auto between_iter_task(Index N = default_N)
    { // transform: fused operations in one lambda, mapping indices to values
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, 3);

        for (auto _ : p_loop_state) {
            V[0] = 0;
            W[0] = 0;

            #pragma omp parallel
            #pragma omp single
            {
                for (Index i = 1; i < V.size(); i++) {
                    #pragma omp task depend(out: V[i])
                    {
                        V[i] = func(i);
                    }

                    #pragma omp task depend(in: V[i-1]) depend(out: W[i])
                    {
                        W[i] = func2(V[i - 1]);
                    }
                }

                #pragma omp taskwait
            }

            p_loop_action();
        }

        // output
        std::copy_n(cbegin(W), default_Nout, std::ostream_iterator<Real>(p_log, ","));
        p_log << '\n';
        // input and output size in bytes
        return std::tuple {N * sizeof(Real), N * sizeof(Real)};
    }
};

template<typename R, typename L>
array<char, 1> Dependency<R, L>::p_one = {0};
} // namespace NUMA_omp
