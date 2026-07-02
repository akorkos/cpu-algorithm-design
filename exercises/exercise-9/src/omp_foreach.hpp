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
#include <random>
#include <execution>
#include <omp.h>
#include "numa_extensions.hpp"
#include <xsimd/xsimd.hpp>

namespace NUMA_omp
{
    using namespace std;
    template <typename R = array<char, 1>, typename L = function<void(void)>>
    class foreach
    {
    private:
        static array<char, 1> p_one;
        stringstream p_log;
        R &p_loop_state;
        L p_loop_action;

    public:
        foreach (R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << fixed << setprecision(2); }
        string get_log() { return p_log.str(); }

        using Index = int;
        using Int = int32_t;
        using Real = double;
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

        static constexpr Index default_nx = 3;
        static constexpr Index default_ny = 5;
        static constexpr Index default_nz = 3;
        static constexpr Index default_Nx = 4;
        static constexpr Index default_Ny = 8;
        static constexpr Index default_Nz = 2;
        static constexpr Index Nexp = 3;

        auto Subrange_interval(Index N = default_N)
        { // for_each interval
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            Real a = 2;
            Real b = 4;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, a, b, N)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < N; i++)
                    {
                        W[i] = (i >= 2 && i < N - 3) ? a * V[i] : b * V[i];
                    }
                }

                p_loop_action();
            }

            p_log << "interval:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N * sizeof(Real)};
        }

        auto Subrange_intervals(Index N = default_N)
        { // for_each intervals
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<pair<Index, Index>> Q(N - 5); // [2,3), [3,4), ...
            transform(mtExec, itSeq(2), itSeq(N - 3), begin(Q), [](Index i)
                      { return pair{i, i + 1}; });

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
                NUMAContainer<pair<Index, Index>> intervals(N - 5);

#pragma omp parallel default(none) shared(V, W, a, intervals)
                {
#pragma omp for simd
#pragma unroll
                    for (Index k = 0; k < N - 5; k++)
                    {
                        intervals[k] = {k + 2, k + 3};
                    }

#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < intervals.size(); i++)
                    {
                        W[intervals[i].first] = a * V[intervals[i].first];
                    }
                }
                p_loop_action();
            }

            p_log << "intervals:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N * sizeof(Real)};
        }

        auto Subrange_singles(Index N = default_N)
        { // for_each single indices - DONT
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Index> Q(N-5);
            copy(mtExec, itSeq(2), itSeq(N-3), begin(Q));

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            Real a = 2;

            // auto vIndices = views::join(indices);
            for (auto _ : p_loop_state)
            {
                NUMAContainer<Index> indices(N - 5); // DONT unless necessary

// for (auto i : {2,3,4,...,N-4}) { W[i] = a * V[i]; }
#pragma omp parallel default(none) shared(V, W, a, indices)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t k = 0; k < N - 5; k++)
                    {
                        indices[k] = k + 2;
                    }

#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < indices.size(); i++)
                    {
                        W[indices[i]] = a * V[indices[i]];
                    }
                }
                p_loop_action();
            }

            p_log << "singles:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N * sizeof(Real)};
        }

        auto Subrange_soa_explicit_stride_domain(Index N = default_N)
        { // for_each SoA(explicit) strided subrange, enlarged domain - PREFERRED
            constexpr Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            Index nind = 1;                              // selected index in [0,n)
            auto divc_Nn = div_ceil(N, n);               // quot*n + rem = N
            auto Nrow = divc_Nn.quot;                    // size of row
            auto Nsup = Nrow * n;                        // superset of indices
            // auto vStride = vSeq_n(nind * Nrow, Nrow, 1); // every n-th element with contiguous accesses
            auto vStride0 = vSeq_n(0 * Nrow, Nrow, 1); 
            auto vStride1 = vSeq_n(1 * Nrow, Nrow, 1); 
            auto vStride2 = vSeq_n(2 * Nrow, Nrow, 1);
    
            NUMAContainer<Real, n> V(Nsup, -1);

#pragma omp parallel default(none) shared(V, Nrow, n)
            {
#pragma omp for simd
#pragma unroll
                for (size_t i = 0; i < V.size(); i++)
                {
                    V[i] = (i % Nrow) * n + (i / Nrow);
                }
            }

            NUMAContainer<Real> W(Nsup, -1);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, a, vStride0, vStride1, vStride2)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < Nrow; i++)
                    {
                        W[vStride0[i]] = V[vStride0[i]];
                        W[vStride1[i]] = a * V[vStride1[i]];
                        W[vStride2[i]] = a * V[vStride2[i]] * V[vStride2[i]];
                    }
                }
                p_loop_action();
            }

            p_log << "stride:";
            for (Index j = 0; j < n; ++j)
                p_log << "\t" << (vSeq_n(j * Nrow, min(Nout, Nrow)) | vWith(W)) << '\n';
            return tuple{N / n * sizeof(Real), N / n * sizeof(Real)};
        }

        auto Reordering_reverse(Index N = 1 << Nexp)
        { // for_each reverse
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            Index Nexp_ = log(N) / log(2);

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(V, W, N, Nexp_)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < N; i++)
                    {
                        int new_i = 0;
                        int temp = i;
                        
                        for (int j = 0; j < Nexp_; j++) 
                        {
                            new_i <<= 1; 
                            new_i |= (temp & 1); 
                            temp >>= 1;              
                        }
                    
                        W[i] = V[new_i];
                    }
                }
                p_loop_action();
            }

            p_log << "reverse:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N * sizeof(Real)};
        }

         auto Reordering_gather_data(Index N = default_N)
        { // for_each gather data
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            NUMAContainer<Real> P(N);
            NUMAContainer<Real> P1(N);
            NUMAContainer<Index> Q(N); 

#pragma omp parallel default(none) shared(P, P1, Q, n, m, N)
            {
#pragma omp for simd
#pragma unroll
                for (size_t i = 0; i < P.size(); i++)
                {
                    P[i] = (i + n) % N;
                    P1[i] = (i + m) % N;
                    Q[i] = P1[P[i]];
                }
            }

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(V, W, Q)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = V[Q[i]];
                    }
                }
                p_loop_action();
            }

            p_log << "gather:\t" << views::take(W, Nout) << '\n';
            return tuple{N * (sizeof(Real) + sizeof(Index)), N * sizeof(Real)};
        }

        auto Reordering_scatter_func_n1(Index N = default_N)
        { // for_each scatter func n -> 1 - DONT
            Index n = default_n;
            Index m = default_m;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, n)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < N; i++)
                    {
                        if (i % n == n - 1){
                            W[i / n] = V[i];
                        }
                    }
                }
                p_loop_action();
            }

            p_log << "scatter:" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N / n * sizeof(Real)};
        }

        auto Reordering_is_permutation(Index N = default_N)
        {
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            std::next_permutation(V.begin(), V.end());

            for(auto _ : p_loop_state)
            {
                std::sort(mtExec, V.begin(), V.end());
                int ans = 0;
                
                #pragma omp parallel for simd reduction(+:ans) default(none) shared(V, N)
                for (size_t i = 0; i < N; i++)
                {
                    ans += (i != V[i]);
                    //if(i!=V[i]) ans++;
                }

                p_log << (ans > 0 ? "failed." : "success.");
                p_loop_action();
            }
                
            return tuple{N * sizeof(Real), N / 3 * sizeof(Real)};
        }
    };
 
    template <typename R, typename L>
    array<char, 1> foreach<R, L>::p_one = {0};
}
