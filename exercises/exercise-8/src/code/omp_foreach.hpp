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

        auto Map_AoS_scalar_scalar(Index N = default_N)
        { // for_each scalar -> scalar
            Index n = 1;
            Index m = n;
            Index Nout = min(N, default_Nout);
            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            NUMAContainer<Real> W(N, -1);
            Real a = 2;
            Real b = 3;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, a, b)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < W.size(); i++)
                    {
                        W[i] = a * V[i] + b * W[i];
                    }
                }
                p_loop_action();
            }

            p_log << "a*v+b*w:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real), N * sizeof(Real)};
        }

        auto Map_AoS_complex_complex(Index N = default_N)
        { // for_each complex -> complex
            Index n = 2;
            Index m = 2;
            Index Nout = min(N, default_Nout);

            NUMAContainer<CReal> V0(itSeq(0), itSeq(N));
            NUMAContainer<CReal> V1(itSeq(0), itSeq(N));
            NUMAContainer<CReal> V2(itSeq(0), itSeq(N));

            NUMAContainer<CReal> W0(N, -1);
            NUMAContainer<CReal> W1(N, -1);
            NUMAContainer<CReal> W2(N, -1);

            CReal a0 = {0, 2};
            CReal a1 = {0,3};
            CReal a2 = {0,5};
            
            CReal b0 = {0,1};
            CReal b1 = {0,7};
            CReal b2 = {0,11};

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, a0, a1, a2, b0, b1, b2)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < W.size(); i++)
                    {
                        W0[i] = a0 * V0[i] + b0 * W0[i];
                        W1[i] = a1 * V1[i] + b1 * W1[i];
                        W2[i] = a2 * V2[i] + b2 * W2[i];
                    }
                }
                p_loop_action();
            }

            p_log << "a*i*v:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(CReal), N * sizeof(CReal)};
        }

        auto Map_AoS_aos_nn_flat(Index N = default_N)
        { // for_each AoS n -> n flat - PREFERRED
            Index n = default_n;
            Index m = n;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(n * N);
#pragma omp parallel default(none) shared(V, n)
            {
#pragma omp for simd
#pragma unroll
                for (size_t i = 0; i < V.size(); i++)
                {
                    V[i] = (i / n) * !(i % n);
                }
            }

            NUMAContainer<Real> W(n * N, -1);

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, N, n)
                {
#pragma omp for simd
#pragma unroll
                    for (size_t i = 0; i < N; i++)
                    {
                        rotate_copy(V.begin() + i * n, V.begin() + i * n + n / 2, V.begin() + i * n + n, W.begin() + i * n);
                    }
                }

                p_loop_action();
            }

            p_log << "rotate:\t" << (W | views::chunk(n) | views::take(Nout)) << '\n';
            return tuple{N * sizeof(Real) * n, N * sizeof(Real) * n};
        }
        
        auto Map_AoS_aos_nm_flat_spantmp(Index N = default_N)
        { // for_each AoS n -> m flat with span tmp - PREFERRED
            Index n = default_n;
            Index m = min(n, default_m);
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(n * N));

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(1, 99);

            NUMAContainer<Real> W(m * N, -1);
            std::transform(mtExec, W.begin(), W.end(), W.begin(), [&gen, &distrib](auto w){ return w=distrib(gen);});

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, N, n, m)
                {
#pragma omp for // naive sort can't be vectorized
#pragma unroll
                    for (size_t i = 0; i < N; i++)
                    {
                        auto tmpSize = n + m;
                        Real tmp_[tmpSize];
                        auto tmp = span{tmp_, static_cast<size_t>(tmpSize)}; // using dynamic n+m
                        copy_n(stExec, begin(V) + (i * n), n, begin(tmp));
                        copy_n(stExec, begin(W) + (i * m), m, begin(tmp) + n);
                        sort(stExec, begin(tmp), begin(tmp) + n, greater());
                        sort(stExec, begin(tmp) + n, begin(tmp) + tmpSize, greater());
                        inplace_merge(stExec, begin(tmp), begin(tmp) + n, begin(tmp) + tmpSize, greater());
                        copy_n(stExec, cbegin(tmp), m, begin(W) + (i * m));
                    }
                }
                p_loop_action();
            }

            p_log << "merge:\t" << (W | views::chunk(m) | views::take(Nout)) << '\n';
            return tuple{N * sizeof(Real) * (n + m), N * sizeof(Real) * m};
        }

        auto Map_AoS_aos_tuple_tuple(Index N = default_N)
        { // for_each AoS tuple -> tuple
            constexpr Index n = 3;
            Index m = n;
            Index Nout = min(N, default_Nout);

            using iElement = tuple<Real,Real>; 
            static_assert(tuple_size<iElement>() == n -1);
            using oElement = tuple<Index, Real, Real>;
            static_assert(tuple_size<oElement>() == n);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(1, 99);
            
            Container<iElement> V(N); 

            transform(mtExec, itSeq(0), itSeq(N), begin(V), [&gen, &distrib](auto i){ 
                auto b = distrib(gen);
                auto c = distrib(gen);
                return iElement{b,c};
            });
  
            NUMAContainer<oElement> V(N);

#pragma omp parallel default(none) shared(V)
            {
#pragma omp for simd
#pragma unroll
                for (Index i = 0; i < V.size(); i++)
                {
                    V[i] = oElement{i, i, i};
                }
            }

            NUMAContainer<oElement> W(N, oElement{-1, -1, -1});
            Real a = 2;

            for (auto _ : p_loop_state)
            {
                #pragma omp parallel default(none) shared(V, W, a)
                {
                    #pragma omp for simd
                    #pragma unroll
                    for (Index i = 0; i < W.size(); i++)
                    {
                        auto [a_,b_] = V[i];
                        Real smallest_, greatest_;
                        Index change = 0;

                        if (a_ < b_) 
                        {
                            change = 1;
                        }

                        get<0>(W[i]) = change;
                        get<1>(W[i]) = change ? b_ : a_ ;
                        get<2>(W[i]) = change ? a_ : b_;
                    }
                }
                p_loop_action();
            }

            p_log << "v,a*v,a*a*v:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(oElement), N * sizeof(oElement)};
        }

         auto Map_AoS_aos_tuple_pair(Index N = default_N)
        { // for_each AoS tuple -> pair
            constexpr Index n = 3;
            constexpr Index m = 2;
            Index Nout = min(N, default_Nout);

            using ElementV = tuple<Index, Real, CReal>;
            static_assert(tuple_size<ElementV>() == n);
            using ElementW = pair<Index, Real>;
            static_assert(tuple_size<ElementW>() == m);
            NUMAContainer<ElementV> V(N);
#pragma omp parallel default(none) shared(V)
            {
#pragma omp for simd
#pragma unroll
                for (Index i = 0; i < V.size(); i++)
                {
                    V[i] = ElementV{i, i, i};
                }
            }
            NUMAContainer<ElementW> W(N, ElementW{-1, -1});
            Real a = 2;

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(V, W, a)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < W.size(); i++)
                    {
                        auto &[wi, wr] = W[i];
                        auto &[vi, vr, vc] = V[i];
                        wi = vi;
                        wr = (CReal{0,a} * vc).imag();
                    }
                }
                p_loop_action();
            }

            p_log << "v,a*i*v:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(ElementV), N * sizeof(ElementW)};
        }

         auto Map_SoA_soa_zip_tuple_tuple(Index N = default_N)
        { // for_each SoA(zip) tuple -> tuple - PREFERRED
            constexpr Index n = 3;
            Index m = n;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Index> V0(itSeq(0), itSeq(N));
            NUMAContainer<Real> V1(itSeq(0), itSeq(N));
            NUMAContainer<Real> V2(itSeq(0), itSeq(N));
            auto V = views::zip(V0, V1, V2);
            using Element = ranges::range_value_t<decltype(V)>;
            static_assert(tuple_size<Element>() == n);
            NUMAContainer<Index> W0(N, -1);
            NUMAContainer<Real> W1(N, -1);
            NUMAContainer<Real> W2(N, -1);
            auto W = views::zip(W0, W1, W2);
            Real a = 2;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, N, a)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < N; i++)
                    {
                        get<0>(W[i]) = get<0>(V[i]);
                        get<1>(W[i]) = a * get<1>(V[i]);
                        get<2>(W[i]) = a * a * get<2>(V[i]);
                    }
                }
                p_loop_action();
            }

            p_log << "v,a*v,a*a*v:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Element), N * sizeof(Element)};
        }

         auto Map_SoA_soa_zip_nm_spantmp(Index N = default_N)
        { // for_each SoA(zip) n -> m with span tmp
            constexpr Index n = 3;
            constexpr Index m = 2;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real, n> V_(n * N, -1);
            auto V = views::zip(vSeq_n(0 * N, N) | vWith(V_), vSeq_n(1 * N, N) | vWith(V_), vSeq_n(2 * N, N) | vWith(V_));
            using ElementV = ranges::range_value_t<decltype(V)>;
            static_assert(tuple_size<ElementV>() == n);
#pragma omp parallel default(none) shared(V, n, N)
            {
#pragma omp for simd
#pragma unroll
                for (Index i = 0; i < N; i++)
                {
                    V[i] = ElementV{i * n + 0, i * n + 1, i * n + 2};
                }
            }
            NUMAContainer<Real, m> W_(m * N, -1);
            auto W = views::zip(vSeq_n(0 * N, N) | vWith(W_), vSeq_n(1 * N, N) | vWith(W_));
            using ElementW = ranges::range_value_t<decltype(W)>;
            static_assert(tuple_size<ElementW>() == m);
            Real a = 2;

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(V, W, N, m, n)
                {
#pragma omp for simd
#pragma unroll
                    for (Index i = 0; i < N; i++)
                    {
                        Real tmp_[n + m];
                        auto tmp = span{tmp_, n + m}; // using dynamic n+m
                        apply([tmp](const auto &...val_s)
                              { Index j=0; (..., (tmp[j++]=val_s)); }, V[i]); // tuple-loop
                        apply([n, tmp](const auto &...val_s)
                              { Index j=n; (..., (tmp[j++]=val_s)); }, W[i]); // tuple-loop
                        sort(stExec, begin(tmp), begin(tmp) + n, greater());
                        inplace_merge(stExec, begin(tmp), begin(tmp) + n, begin(tmp) + n + m, greater());
                        apply([tmp](auto &...val_s)
                              { Index j=0; (..., (val_s=tmp[j++])); }, W[i]); // tuple-loop
                    }
                }
                p_loop_action();
            }

            p_log << "merge:\t" << views::take(W, Nout) << '\n';
            return tuple{N * sizeof(Real) * (n + m), N * sizeof(Real) * m};
        }

        auto Map_SoA_soa_explicit_nm_flat_spantmp(Index N = default_N)
        { // for_each SoA(explicit) n -> m flat with span tmp - PREFERRED
            constexpr Index n = default_n;
            constexpr Index m = min(n, default_m);
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real, n> V(n * N, -1);

#pragma omp parallel default(none) shared(V, N, n)
            {
#pragma omp for simd
#pragma unroll
                for (Index i = 0; i < N * n; i++)
                {
                    V[i] = (i % N) * n + (i / N);
                }
            }

            NUMAContainer<Real, m> W(m * N, -1);

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(V, W, N, m, n)
                {
#pragma omp for // no simd for sort
#pragma unroll
                    for (Index i = 0; i < N; i++)
                    {
                        auto tmpSize = n + m;
                        Real tmp_[n + m];
                        auto tmp = span{tmp_, static_cast<size_t>(n + m)}; // using dynamic n+m
                        auto Vi = vSeq_n(i, n * N, N) | vWith(cbegin(V));  // iterable zip
                        auto Wi = vSeq_n(i, m * N, N) | vWith(begin(W));   // iterable zip
                        for (int z = 0; z < n; z++)
                        {
                            tmp[z] = Vi[z];
                        }
                        for (int z = 0; z < m; z++)
                        {
                            tmp[z + n] = Wi[z];
                        }

                        sort(stExec, begin(tmp), begin(tmp) + n, greater());
                        inplace_merge(stExec, begin(tmp), begin(tmp) + n, begin(tmp) + n + m, greater());
                        for (int z = 0; z < m; z++)
                        {
                            Wi[z] = tmp[z];
                        }
                    }
                }
                p_loop_action();
            }

            p_log << "merge:";
            for (Index j = 0; j < m; ++j)
                p_log << "\t" << (W | views::drop(j * N) | views::take(Nout)) << '\n';
            return tuple{N * sizeof(Real) * (n + m), N * sizeof(Real) * m};
        }
    };

    template <typename R, typename L>
    array<char, 1> foreach<R, L>::p_one = {0};
}
