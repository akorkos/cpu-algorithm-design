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
#include <xsimd/xsimd.hpp>

namespace NUMA_omp {
  using namespace std;

  template<typename R = array<char, 1>, typename L = function<void(void)>>
  class Stencil {
    private:
      static array<char, 1> p_one;
      stringstream          p_log;
      R                    &p_loop_state;
      L                     p_loop_action;

    public:
      Stencil(
        R &loop_state = p_one, L loop_action = []() {})
          : p_loop_state(loop_state), p_loop_action(loop_action) {
        p_log << fixed << setprecision(2);
      }

      string get_log() { return p_log.str(); }

      using Index     = int;
      using Int       = int32_t;
      using Real      = float;
      using LongReal  = long double;
      using CReal     = complex<Real>;
      using LongCReal = complex<LongReal>;
      template<typename T>
      using Container =
        vector<T, numa::no_init_allocator<T>>; // TODO use no-init-allocator
                                               // this will cause old code crash
      // using Container = deque<T, allocator<T>>; // TODO use no-init-allocator
      template<typename T, int Fold = 1>
      using NUMAContainer = numa_extensions<Container<T>, Fold>;
      static constexpr auto stExec =
        execution::unseq;     // single-threaded execution policy
      static constexpr auto mtExec =
        execution::par_unseq; // multi-threaded execution policy

      static constexpr Index default_n    = 3;
      static constexpr Index default_m    = 2;
      static constexpr Index default_N    = 20;
      static constexpr Index default_Nout = 10;

      static constexpr Index default_nx = 3;
      static constexpr Index default_ny = 5;
      static constexpr Index default_nz = 3;
      static constexpr Index default_Nx = 4;
      static constexpr Index default_Ny = 8;
      static constexpr Index default_Nz = 2;

      auto Map_AoS_scalar_scalar(
        Index N = default_N) { // for_each scalar -> scalar
        Index               n    = 1;
        Index               m    = n;
        Index               Nout = min(N, default_Nout);
        NUMAContainer<Real> V(itSeq(0), itSeq(N));
        NUMAContainer<Real> W(N, -1);
        Real                a = 2;

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, W, a)
          {
#pragma omp for simd
#pragma unroll
            for (size_t i = 0; i < W.size(); i++) { W[i] = a * V[i]; }
          }
          p_loop_action();
        }

        p_log << "a*v:\t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real), N * sizeof(Real)};
      }

      auto Stencil_1d_dirichlet_index(
        Index Ncur = default_N) { // for_each stencil 1D, Dirichlet, exact index
                                  // computation
        Index nx   = default_nx;
        Index ny   = 1;
        Index nz   = 1;
        Index Nx   = Ncur;
        Index Ny   = 1;
        Index Nz   = 1;
        Index Nout = default_Nout;

        auto                N0       = Nx;
        auto                n0       = nx;
        auto                Nsize    = N0;
        auto                nsize    = n0;
        Real                boundary = 0;
        NUMAContainer<Real> V(itSeq(0), itSeq(Nsize));
        NUMAContainer<Real> W(Nsize, -1);
        NUMAContainer<Real> S(nsize, 1);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, W, S, boundary, n0, N0, Nsize)
          {
#pragma omp for simd
#pragma unroll
            for (Index i = 0; i < Nsize; i++) {
              Real sum = 0;
              for (Index j0 = 0; j0 < n0; ++j0) {
                auto ind0 = i + (j0 - n0 / 2);
                auto lim0 = (ind0 < 0) || (ind0 >= N0);
                auto v    = (lim0) ? boundary : V[ind0];
                sum += S[j0] * v;
              }
              W[i] = sum;
            }
          }
          p_loop_action();
        }

        p_log << "stencil 1D:\t" << views::take(W, Nout) << '\n';
        return tuple{Nx * sizeof(Real), Nx * sizeof(Real)};
      }

      auto Stencil_1d_neumann_index(
        Index Ncur =
          default_N) { // for_each stencil 1D, Neumann, exact index computation
        Index nx   = default_nx;
        Index ny   = 1;
        Index nz   = 1;
        Index Nx   = Ncur;
        Index Ny   = 1;
        Index Nz   = 1;
        Index Nout = default_Nout;

        auto                N0    = Nx;
        auto                n0    = nx;
        auto                Nsize = N0;
        auto                nsize = n0;
        NUMAContainer<Real> V(itSeq(0), itSeq(Nsize));
        NUMAContainer<Real> W(Nsize, -1);
        NUMAContainer<Real> S(nsize, 1);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, W, S, n0, N0, Nsize)
          {
#pragma omp for simd
#pragma unroll
            for (Index i = 0; i < Nsize; i++) {
              Real sum = 0;
              for (Index j0 = 0; j0 < n0; ++j0) {
                auto ind0 = i + (j0 - n0 / 2);
                ind0      = clamp(ind0, 0, N0 - 1);
                sum += S[j0] * V[ind0];
              }
              W[i] = sum;
            }
          }
          p_loop_action();
        }

        p_log << "stencil 1D:\t" << views::take(W, Nout) << '\n';
        return tuple{Nx * sizeof(Real), Nx * sizeof(Real)};
      }

      auto Stencil_1d_dirichlet_domain(
        Index Ncur =
          default_N) { // for_each stencil 1D, Dirichlet, enlarged domain
        Index nx   = default_nx;
        Index ny   = 1;
        Index nz   = 1;
        Index Nx   = Ncur;
        Index Ny   = 1;
        Index Nz   = 1;
        Index Nout = default_Nout;

        auto                N0       = Nx;
        auto                n0       = nx;
        auto                Nsize    = N0 + n0;
        auto                nsize    = n0;
        Real                boundary = 0;
        NUMAContainer<Real> V(Nsize);

#pragma omp parallel default(none) shared(V, n0, N0, boundary)
        {
#pragma omp for
#pragma unroll
          for (Index i = 0; i < V.size(); i++) {
            if (i < n0 / 2 || i >= N0 + n0 / 2) {
              V[i] = boundary;
            } else {
              V[i] = i - n0 / 2;
            }
          }
        }
        NUMAContainer<Real> W(Nsize, -1);
        NUMAContainer<Real> S(nsize, 1);

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none) shared(V, W, S, boundary, n0, N0)
          {
#pragma omp for simd
#pragma unroll
            for (Index i = n0 / 2; i < N0 + n0 / 2; i++) {
              Real sum = 0;
              for (Index j0 = 0; j0 < n0; ++j0) {
                sum += S[j0] * V[i + (j0 - n0 / 2)];
              }
              W[i] = sum;
            }
          }
          p_loop_action();
        }

        p_log << "stencil 1D:\t" << views::take(W, Nout) << '\n';
        return tuple{Nx * sizeof(Real), Nx * sizeof(Real)};
      }

      auto Stencil_2d_dirichlet_index(
        Index Ncur = default_N) { // for_each stencil 2D, Dirichlet, exact index
                                  // computation
        constexpr Index nx = default_nx;
        constexpr Index ny = default_ny;
        Index           nz = 1;
        auto            Na = pow(Ncur / Real{default_Nx * default_Ny}, 1 / 2.0);
        Index           Nx = ceil(default_Nx * Na);
        Index           Ny = ceil(default_Ny * Na);
        Index           Nz = 1;
        Index           Nout = default_Nout;

        auto           N        = array{Nx, Ny};
        constexpr auto n        = array{nx, ny};
        auto           Nsize    = N[0] * N[1];
        auto           nsize    = n[0] * n[1];
        Real           boundary = 0;
        auto vN2D     = views::cartesian_product(vSeq(0, N[1]), vSeq(0, N[0]));
        using Element = ranges::range_value_t<decltype(vN2D)>;
        static_assert(tuple_size<Element>() == N.size());
        auto Nt2s = [N](const Element i2D) {
          return get<0>(i2D) + N[0] * get<1>(i2D);
        };
        auto Ns2t = [N](const Index i) { return Element{i % N[0], i / N[0]}; };
        auto vn2D = views::cartesian_product(vSeq(0, n[0]), vSeq(0, n[1]));
        auto nt2s = [n](const Element i2D) {
          return get<0>(i2D) + n[0] * get<1>(i2D);
        };
        auto ns2t = [n](const Index i) { return Element{i % n[0], i / n[0]}; };

        NUMAContainer<Real> V(Nsize, -1);
#pragma omp parallel default(none) shared(V, vN2D, Nt2s)
        {
#pragma omp for
#pragma unroll
          for (auto i2D : vN2D) {
            auto [i1, i0]     = i2D;
            V[Nt2s({i0, i1})] = i0 + i1 * 10;
          }
        }

        NUMAContainer<Real> W(Nsize, -1);

        NUMAContainer<Real> S(nsize);
#pragma omp parallel default(none) shared(S, vn2D, nt2s, n)
        {
#pragma omp for
#pragma unroll
          for (auto i2D : vn2D) {
            auto [i0, i1] = i2D;
            i0 -= n[0] / 2;
            i1 -= n[1] / 2;
            S[nt2s(i2D)] = ((i0 == 0) && (i1 == 0));
          }
        }

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none)                                             \
  shared(S, V, W, vN2D, nt2s, Nt2s, n, boundary, N)
          {
#pragma omp for simd
#pragma unroll
            for (auto i2D : vN2D) {
              auto [i1, i0] = i2D;
              Real sum      = 0;
#pragma unroll(n[1])
              for (Index j1 = 0; j1 < n[1]; ++j1) {
                auto ind1 = i1 + (j1 - n[1] / 2);
                auto lim1 = (ind1 < 0) || (ind1 >= N[1]);
#pragma unroll
                for (Index j0 = 0; j0 < n[0]; ++j0) {
                  auto ind0 = i0 + (j0 - n[0] / 2);
                  auto lim0 = (ind0 < 0) || (ind0 >= N[0]);
                  auto v    = (lim0 || lim1) ? boundary : V[Nt2s({ind0, ind1})];
                  sum += S[nt2s({j0, j1})] * v;
                }
              }

              W[Nt2s({i0, i1})] = sum;
            }
          }
          p_loop_action();
        }

        p_log << "stencil 2D:\n";
        for (Index i1 = 0; i1 < min(Nout, N[1]); ++i1) {
          p_log << '\t'
                << (W | views::drop(Nt2s({0, i1})) |
                    views::take(min(Nout, N[0])))
                << '\n';
        }
        return tuple{Nx * Ny * sizeof(Real), Nx * Ny * sizeof(Real)};
      }

      auto Stencil_2d_dirichlet_index_explicit(
        Index Ncur = default_N) { // for_each stencil 2D, Dirichlet, exact index
                                  // computation
        constexpr Index nx = default_nx;
        constexpr Index ny = default_ny;
        Index           nz = 1;
        auto            Na = pow(Ncur / Real{default_Nx * default_Ny}, 1 / 2.0);
        Index           Nx = ceil(default_Nx * Na);
        Index           Ny = ceil(default_Ny * Na);
        Index           Nz = 1;
        Index           Nout = default_Nout;

        const auto     N        = array{Nx, Ny};
        constexpr auto n        = array{nx, ny};
        auto           Nsize    = N[0] * N[1];
        auto           nsize    = n[0] * n[1];
        Real           boundary = 0;
        auto vN2D     = views::cartesian_product(vSeq(0, N[0]), vSeq(0, N[1]));
        using Element = ranges::range_value_t<decltype(vN2D)>;
        static_assert(tuple_size<Element>() == N.size());
        auto Nt2s = [N](const Element i2D) {
          return get<0>(i2D) + N[0] * get<1>(i2D);
        };
        auto Ns2t = [N](const Index i) { return Element{i % N[0], i / N[0]}; };
        auto vn2D = views::cartesian_product(vSeq(0, n[0]), vSeq(0, n[1]));
        auto nt2s = [n](const Element i2D) {
          return get<0>(i2D) + n[0] * get<1>(i2D);
        };
        auto ns2t = [n](const Index i) { return Element{i % n[0], i / n[0]}; };

        NUMAContainer<Real> V(Nsize, -1);
#pragma omp parallel default(none) shared(V, vN2D, Nt2s, nx, ny, N, boundary)
        {
#pragma omp for
#pragma unroll
          for (auto i2D : vN2D) {
            auto [i0, i1] = i2D;

            auto lim1    = (i0 < nx / 2) || (i0 >= N[0] - nx / 2);
            auto lim2    = (i1 < ny / 2) || (i1 >= N[1] - ny / 2);
            auto v       = (lim1 || lim2) ? boundary : i0 + i1 * 10;
            V[Nt2s(i2D)] = v;
          }
        }
        NUMAContainer<Real> W(Nsize, -1);

        NUMAContainer<Real> S(nsize);
#pragma omp parallel default(none) shared(S, vn2D, nt2s, n)
        {
#pragma omp for
#pragma unroll
          for (auto i2D : vn2D) {
            auto [i0, i1] = i2D;
            i0 -= n[0] / 2;
            i1 -= n[1] / 2;
            S[nt2s(i2D)] = ((i0 == 0) && (i1 == 0));
          }
        }

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none)                                             \
  shared(S, V, W, vN2D, nt2s, Nt2s, n, boundary, Nsize, N, Nx, Ny, nx, ny)
          {
#pragma omp for
#pragma unroll
            for (Index z = ny / 2; z < Ny - ny / 2; z++) {
#pragma omp simd
              for (Index j = nx / 2; j < Nx - nx / 2; j++) {
                Real sum = 0;

                auto ind0 = j - nx / 2; // need to be here
                // auto lim0 = (ind0 < 0) || (ind0>= N[0]); // putting this into
                // the inner loop halve the throughput
                for (Index j1 = 0; j1 < ny; ++j1) {
                  auto ind1 = z - ny / 2 + j1;
                  // auto lim1 = (ind1 < 0) || (ind1 >= N[1]);
                  auto Vind = N[0] * ind1;
                  auto Sind = nx * j1;
                  for (Index j0 = 0; j0 < nx; ++j0) {
                    // auto v = (lim0 || lim1) ? boundary : V[ind0 + Vind + j0];
                    sum += S[j0 + n[0] * j1] * V[ind0 + Vind + j0];
                  }
                }
                W[z * Nx + j] = sum;
              }
            }
          }
          p_loop_action();
        }

        p_log << "stencil 2D:\n";
        for (Index i1 = 0; i1 < min(Nout, N[1]); ++i1) {
          p_log << '\t'
                << (W | views::drop(Nt2s({0, i1})) |
                    views::take(min(Nout, N[0])))
                << '\n';
        }
        return tuple{Nx * Ny * sizeof(Real), Nx * Ny * sizeof(Real)};
      }

      auto Stencil_3d_dirichlet_index_explicit(
        Index Ncur = default_N) { // for_each stencil 3D, Dirichlet, exact index
                                  // computation
        constexpr Index nx = 3;
        constexpr Index ny = 5;
        constexpr Index nz = 3;
        auto            Na =
          pow(Ncur / Real{default_Nx * default_Ny * default_Nz}, 1 / 3.0);
        Index Nx       = ceil(default_Nx * Na);
        Index Ny       = ceil(default_Ny * Na);
        Index Nz       = ceil(default_Nz * Na);
        Index Nout     = default_Nout;
        auto  N        = array{Nx, Ny, Nz};
        auto  n        = array{nx, ny, nz};
        auto  Nsize    = N[0] * N[1] * N[2];
        auto  nsize    = n[0] * n[1] * N[2];
        Real  boundary = 0;
        auto  vN3D =
          views::cartesian_product(vSeq(0, N[0]), vSeq(0, N[1]), vSeq(0, N[2]));
        using Element = ranges::range_value_t<decltype(vN3D)>;
        static_assert(tuple_size<Element>() == N.size());
        auto Nt2s = [N](const Element i3D) {
          return get<0>(i3D) + N[0] * (get<1>(i3D) + N[1] * get<2>(i3D));
        };
        auto Ns2t = [N](const Index i) {
          return tuple{i % N[0], (i / N[0]) % N[1], (i / N[0]) / N[1]};
        };
        auto vn3D =
          views::cartesian_product(vSeq(0, n[0]), vSeq(0, n[1]), vSeq(0, n[2]));
        auto nt2s = [n](const Element i3D) {
          return get<0>(i3D) + n[0] * (get<1>(i3D) + n[1] * get<2>(i3D));
        };
        auto ns2t = [n](const Index i) {
          return tuple{i % n[0], (i / n[0]) % n[1], (i / n[0]) / n[1]};
        };

        NUMAContainer<Real> V(Nsize);
#pragma omp parallel default(none) shared(V, vN3D, Nt2s)
        {
#pragma omp for
#pragma unroll
          for (auto i3D : vN3D) {
            auto [i0, i1, i2] = i3D;
            V[Nt2s(i3D)]      = i0 + i1 * 10 + i2 * 100;
          }
        }
        NUMAContainer<Real> W(Nsize);
        NUMAContainer<Real> S(nsize);
#pragma omp parallel default(none) shared(V, vn3D, nt2s, S, n)
        {
#pragma omp for
#pragma unroll
          for (auto i3D : vn3D) {
            auto [i0, i1, i2] = i3D;
            i0 -= n[0] / 2;
            i1 -= n[1] / 2;
            i2 -= n[2] / 2;
            S[nt2s(i3D)] = ((i0 == 0) && (i1 == 0) && (i2 == 0));
          }
        }

        for (auto _ : p_loop_state) {
#pragma omp parallel default(none)                                             \
  shared(V, S, W, vN3D, nt2s, Nt2s, boundary, n, N, Nx, Ny, Nz, nx, ny, nz)
          {
#pragma omp for collapse(2)
            for (Index i2 = nz / 2; i2 < Nz - nz / 2; i2++) {
              for (Index i1 = ny / 2; i1 < Ny - ny / 2; i1++) {
#pragma omp simd
#pragma unroll(16)
                for (Index i0 = nx / 2; i0 < Nx - nx / 2; i0++) {
                  // auto [i0, i1, i2] = i3D;
                  auto ind0 = i0 - nx / 2;
                  Real sum  = 0;
                  // #pragma unroll(nz)
                  for (Index j2 = 0; j2 < nz; ++j2) {
                    auto ind2 = (i2 + (j2 - nz / 2)) * Nx * Ny;
                    // auto lim2 = (ind2 < 0) || (ind2 >= N[2]);
                    for (Index j1 = 0; j1 < ny; ++j1) {
                      auto ind1 = ind2 + (i1 + (j1 - ny / 2)) * Nx;
                      auto ind3 = nx * (j1 + ny * j2);
                      // auto lim1 = (ind1 < 0) || (ind1 >= N[1]);
                      for (Index j0 = 0; j0 < nx; ++j0) {
                        // auto lim0 = (ind0 < 0) || (ind0 >= N[0]);
                        // auto v = (lim0 || lim1 || lim2) ? boundary : V[ind0 +
                        // Nx * ind1 + Ny * Nx * ind2 + j0]; sum += S[nt2s({j0,
                        // j1, j2})] * v;
                        sum += S[j0 + ind3] * V[ind0 + ind1 + j0];
                      }
                    }
                  }
                  W[i0 + Nx * (i1 + i2 * Ny)] = sum;
                }
              }
            }
          }
          p_loop_action();
        }

        p_log << "stencil 3D:\n";
        for (Index i2 = 0; i2 < min(Nout, N[2]); ++i2, p_log << '\n') {
          for (Index i1 = 0; i1 < min(Nout, N[1]); ++i1) {
            p_log << '\t'
                  << (W | views::drop(Nt2s({0, i1, i2})) |
                      views::take(min(Nout, N[0])))
                  << '\n';
          }
        }
        return tuple{Nx * Ny * Nz * sizeof(Real), Nx * Ny * Nz * sizeof(Real)};
      }
  };

  template<typename R, typename L> array<char, 1> Stencil<R, L>::p_one = {0};
} // namespace NUMA_omp
