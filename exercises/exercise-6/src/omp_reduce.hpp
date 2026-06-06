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

template <typename T>
void print_batch(const xsimd::batch<T> &b, const std::string &label = "")
{
    alignas(xsimd::default_arch::alignment()) T arr[xsimd::batch<T>::size];
    b.store_aligned(arr);
    if (!label.empty())
        std::cout << label << ": ";
    std::cout << "[ ";
    for (std::size_t i = 0; i < xsimd::batch<T>::size; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << "]\n";
}
namespace NUMA_omp
{
    using namespace std;
    #pragma declare simd
    double compute_distance(double x, double y){
        return x*x + y*y;
    }
    template <typename R = array<char, 1>, typename L = function<void(void)>>
    class Reduce
    {
    private:
        static array<char, 1> p_one;
        stringstream p_log;
        R &p_loop_state;
        L p_loop_action;

    public:
        Reduce(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << fixed << setprecision(2); }
        string get_log() { return p_log.str(); }

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

        auto reduce_scalar_small(Index N = default_N)
        { // reduce scalar on small or few values
            Index n = 1;
            Index m = n;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            Real res = 0;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for simd reduction(+ : res)
                    for (auto i = cbegin(V); i < cend(V); i++)
                    {
                        res += *i;
                    }
                }

                p_loop_action();
            }

            p_log << "sum:\t" << res << '\n';
            return tuple{N * sizeof(Real), 0};
        }

        auto reduce_scalar(Index N = default_N)
        { // reduce scalar
            Index n = 1;
            Index m = n;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            LongReal init = 0;
            LongReal res = 0;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for simd reduction(+ : res)
                    for (auto i = cbegin(V); i < cend(V); i++)
                    {
                        res += *i;
                    }
                }
                p_loop_action();
            }

            p_log << "sum:\t" << res << '\n';
            return tuple{N * sizeof(Real), 0};
        }

        auto reduce_complex_prod(Index N = default_N)
        { // reduce complex product
            Index n = 2;
            Index m = n;
            Index Nout = min(N, default_Nout);
            NUMAContainer<CReal> V(N, CReal{0, 1});
            CReal res = 1;
#pragma omp declare reduction(* : CReal : omp_out *= omp_in) initializer(omp_priv = CReal{1.0, 0.0})
            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for simd reduction(* : res)
                    for (auto i = cbegin(V); i < cend(V); i++)
                    {
                        res *= *i;
                    }
                }
                p_loop_action();
            }

            p_log << "prod:\t" << res << '\n';
            return tuple{N * sizeof(CReal), 0};
        }

        auto reduce_scalar_norm2(Index N = default_N)
        { // reduce scalar norm2
            Index n = 1;
            Index m = n;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(N, 1);
            LongReal init = 0;
            LongReal res = 0;

            for (auto _ : p_loop_state)
            {
#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for simd reduction(+ : res)
                    for (size_t i = 0; i < V.size(); i++)
                    {
                        res += V[i] * V[i];
                    }
                }
                p_loop_action();
            }
            res = sqrt(res);

            p_log << "norm2:\t" << res << '\n';
            return tuple{N * sizeof(Real), 0};
        }

        auto reduce_n1(Index N = default_N)
        { // reduce n(N) -> 1
            constexpr Index n = 3;
            Index m = 1;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V0(N, 1);
            NUMAContainer<Real> V1(N, 1);
            NUMAContainer<Real> V2(N, 1);
            auto V = views::zip(V0, V1, V2);

            using ElementV = ranges::range_value_t<decltype(V)>;
            static_assert(tuple_size<ElementV>() == n);
#pragma omp parallel default(none) shared(V)
            {
#pragma omp for simd
                for (size_t i = 0; i < V.size(); i++)
                {
                    V[i] = ElementV{1 * i, 2 * i, 3 * i};
                }
            }
            LongReal init = 0;
            LongReal res = 0;

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for simd reduction(+ : res)
                    for (size_t i = 0; i < V.size(); i++)
                    {
                        auto [v0, v1, v2] = V[i];
                        res += (i % 3 > 0) ? ((i % 3 == 2) ? v2 : v1) : v0;
                    }
                }
                p_loop_action();
            }

            p_log << "select:\t" << res << '\n';
            return tuple{N * sizeof(ElementV), 0};
        }

        auto reduce_1m(Index N = default_N)
        { // reduce 1(N) -> m
            Index n = 1;
            constexpr Index m = 3;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V(itSeq(0), itSeq(N));
            using ElementW = tuple<Real, Real, LongReal>;
            static_assert(tuple_size<ElementW>() == m);
            ElementW init{numeric_limits<Real>::lowest(), numeric_limits<Real>::max(), 0}; // lowest() != min()
            ElementW res{0, 0, 0};

#pragma omp declare reduction(elementreduction:ElementW : omp_out = ElementW{max(get<0>(omp_in), get<0>(omp_out)), min(get<1>(omp_in), get<1>(omp_out)), (get<2>(omp_in) + get<2>(omp_out))}) initializer(omp_priv = ElementW{numeric_limits<Real>::lowest(), numeric_limits<Real>::max(), 0})

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for reduction(elementreduction : res)
                    for (size_t i = 0; i < V.size(); i++)
                    {
                        auto tempv = ElementW{V[i], V[i], V[i]};
                        res = ElementW{max(get<0>(tempv), get<0>(res)), min(get<1>(tempv), get<1>(res)), (get<2>(tempv) + get<2>(res))};
                    }
                }
                p_loop_action();
            }

            p_log << "max,min,+:\t" << res << '\n';
            return tuple{N * sizeof(Real), 0};
        }

        auto
        reduce_nm(Index N = default_N)
        { // reduce n(N) -> m
            constexpr Index n = 3;
            constexpr Index m = 4;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V0(N, 1);
            NUMAContainer<Real> V1(N, 1);
            NUMAContainer<Real> V2(N, 1);
            auto V = views::zip(V0, V1, V2);
            using ElementV = ranges::range_value_t<decltype(V)>;
            static_assert(tuple_size<ElementV>() == n);
#pragma omp parallel default(none) shared(V)
            {
#pragma omp for simd
                for (size_t i = 0; i < V.size(); i++)
                {
                    V[i] = ElementV{1 * i, 2 * i, 3 * i};
                }
            }
            using ElementW = tuple<Real, Real, LongReal, LongReal>;
            static_assert(tuple_size<ElementW>() == m);
            ElementW init{numeric_limits<Real>::lowest(), numeric_limits<Real>::max(), 0, 1}; // lowest() != min()
            ElementW res{0, 0, 0, 0};
#pragma omp declare reduction(elementreduction:ElementW : omp_out = ElementW{max(get<0>(omp_in), get<0>(omp_out)), min(get<1>(omp_in), get<1>(omp_out)), (get<2>(omp_in) + get<2>(omp_out)), (get<3>(omp_in) * get<3>(omp_out))}) initializer(omp_priv = ElementW{numeric_limits<Real>::lowest(), numeric_limits<Real>::max(), 0, 1})
            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(res, V)
                {
#pragma omp for reduction(elementreduction : res)
                    for (size_t i = 0; i < V.size(); i++)
                    {
                        auto [v0, v1, v2] = V[i];
                        auto s = (i % 3 > 0) ? ((i % 3 == 2) ? v2 : v1) : v0;
                        res = ElementW{
                            max(s, get<0>(res)),
                            min(s, get<1>(res)),
                            (s + get<2>(res)),
                            (s * get<3>(res))};
                    }
                }
                p_loop_action();
            }

            p_log << "max,min,+,*:\t" << res << '\n';
            return tuple{N * sizeof(ElementV), 0};
        }

        // Appendix -----------------------------------------------------------------------------------

        auto reduce_21(Index N = default_N)
        { // reduce 2(N) -> 1
            Index n = 2;
            Index m = 1;
            Index Nout = min(N, default_Nout);

            NUMAContainer<Real> V0(N, 1);
            NUMAContainer<Real> V1(N, 2);
            LongReal init = 0;
            LongReal res = 0;

            for (auto _ : p_loop_state)
            {

#pragma omp parallel default(none) shared(res, V0, V1)
                {
#pragma omp for reduction(+ : res)
                    for (size_t i = 0; i < V0.size(); i++)
                    {
                        res += V0[i] * V1[i];
                    }
                }
                p_loop_action();
            }

            p_log << "inner:\t" << res << '\n';
            return tuple{N * sizeof(Real) * n, 0};
        }
        auto reduce_pi_grid(Index N = default_N)
        {
            Index num_steps = N;
            int height = std::sqrt(N);
            int width = height;
            num_steps = height * width;
            double pi = 0.0;
            int sum = 0;
            double step = 1.0 / static_cast<double>(height);

            // Generate two random numbers

            for (auto _ : p_loop_state)
            {
                sum = 0.0;
#pragma omp parallel default(none) shared( step, num_steps, height, width) reduction(+ : sum)
                {
#pragma omp for 
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; ++i)
                        {
                            double x = i * step;
                            double y = j * step;
                            sum += floor(compute_distance(x,y));
                        }
                    }
                }
            }
            pi = (N-static_cast<double>(sum)) / N * 4;
            p_log << "pi:\t" << pi << '\n';
            return tuple{long(N) * 6, 0};
        }

        auto reduce_pi_rand(Index N = default_N)
        {
            Index num_steps = N;
            int height = std::sqrt(N);
            int width = height;
            num_steps = height * width;
            double pi = 0.0;
            double sum = 0;
            for (auto _ : p_loop_state)
            {
                sum = 0.0;
#pragma omp parallel default(none) shared(pi, height, width) reduction(+ : sum)
                {
                    uniform_real_distribution dist{0., 1.};
                    default_random_engine rng;
                    rng.seed(0xAAAA);
#pragma omp for schedule(static)
                    for (int j = 0; j < height; j++)
                    {
                        for (Index i = 0; i < width; i++)
                        {
                            int index = j*width + i;
                            rng.seed(hash<Index>{}(index));

                            double x1 = dist(rng);
                            double y1 = dist(rng);
                            sum += floor(x1 * x1 + y1 * y1);
                        }
                    }
                }
            }
            pi = (N - sum) / N * 4;
            p_log << "pi:\t" << pi << '\n';
            return tuple{long(N) * 6, 0};
        }
        auto reduce_boundingbox(Index N = default_N)
        {
            using points = pair<Real, Real>;
            using twoPoints = pair<points, points>;
            NUMAContainer<points> V(N, points{0, 0});
            std::srand(unsigned(std::time(nullptr)));
#pragma omp parallel default(none) shared(V)
            {
                // Thread-local RNG setup
                std::random_device rd;
                std::mt19937 gen(rd() + omp_get_thread_num()); // seed uniquely per thread
                std::uniform_real_distribution<> distrib1(-1, 3);
                std::uniform_real_distribution<> distrib2(-2, 4);

#pragma omp for
                for (size_t i = 0; i < V.size(); ++i)
                {
                    V[i] = points{distrib1(gen), distrib2(gen)};
                }
            }
            int randindex1 = std::rand() % N;
            int randindex2 = std::rand() % N;
            while (randindex1 == randindex2)
                randindex2 = std::rand() % N;
            V[std::rand() % N] = points{-1, 4};
            V[std::rand() % N] = points{3, -2};
            // ---------------------------------------
            // your code start here



            p_log << "lowerleft:\t" << res.first.first << "," << res.first.second << "\tupperright:\t" << res.second.first << "," << res.second.second << '\n';
            return tuple{N * sizeof(Real), 0};
        }
    };
    template <typename R, typename L>
    array<char, 1> Reduce<R, L>::p_one = {0};
}
