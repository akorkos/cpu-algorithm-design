#import "@preview/zebraw:0.6.3": *
#show: zebraw

#set text(font: "New Computer Modern")
#set page(footer: context [
  #h(1fr)
  #counter(page).display(
    "1/1",
    both: true,
  )
])

#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  body
}

//#show figure.caption: set align(center)
#show figure.where(kind: raw): set align(start)

#show table.cell.where(y: 0): set text(weight: "bold")

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

#show figure: set block(breakable: true)

#set heading(outlined: false)

#set math.equation(numbering: "(1)", supplement: [])

#block(
  grid(
    columns: (1fr, auto),
    column-gutter: 1em,
    align: (left, left),
    [
      #block[
        #heading[]
        #text(size: 1.1em, weight: "semibold")[]
        #v(0.5em)
        *Alexandros Korkos* (bn358)\
        #link("mailto:alexandros.korkos@stud.uni-heidelberg.de", "alexandros.korkos[at]stud.uni-heidelberg.de") \
        MSc Computer Engineering \
        Heidelberg University
      ]
    ],
    [
      #image("images/hd_logo.png", width: 4.5cm)
    ],
  ),
)

#align(center)[
  #text(size: 1.8em, weight: "bold")[CPU Algorithm Design]
  #v(0.4em)
  #text(size: 1.2em, weight: "semibold")[Exercise 6]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

= Open MP scheduler 

The `transform_mandelbrot_fixed` is around 10 times faster than `transform_mandelbrot` in all schedulers (default, dynamic, static) probably, because the fixed variant allows points to exit early once they escape the set, while the non-fixed version runs every point to maxIter regardless. This is the classic Mandelbrot problem: an iterative process that doesn't know when to stop, so points inside the set cost maxIter iterations while points outside cost much less, introducing a strong workload imbalance. For the non-fixed version, all three schedulers run within 2 GB/s of each other since the uniform (maxIter-for-everyone) workload balances naturally regardless of scheduling. For the fixed version, this imbalance matters: with static scheduling, threads assigned "expensive" chunks (near the set) sit idling while others finish early, so dynamic wins ($approx 1490$ GB/s) by redistributing work as threads free up, static is close behind ($approx 1442$ GB/s), and default is slowest ($approx 1247$ GB/s), probably falling back to a static-like policy that cannot adapt.

= Calculation of $pi$

With the following implementation @lst:pi-grid, we get a avg througput of $2007.35 "GFLOPs"/s$ (if read correctly). This means a huge improvement compared to the initial implementation and much more then the $700 "GFLOPs"/s$.

#figure(
```cpp
auto reduce_pi_grid(Index N = default_N)
{
  Int height = static_cast<Int>(std::sqrt(static_cast<double>(N)));
  Int width  = height;
  Index num_steps = static_cast<Index>(height) * width;
  Real step_f = 1.0f / static_cast<Real>(height);
  Int sum; 

  for (auto _ : p_loop_state) {
    sum = 0;

    #pragma omp parallel default(none) shared(step_f, height, width) reduction(+:sum)
    {
      #pragma omp for
      for (Index j = 0; j < height; j++) {
        Int row_sum = 0;

        Real y2 = (j * step_f) * (j * step_f); 

        #pragma omp simd reduction(+:row_sum)
        #pragma unroll
        for (Index i = 0; i < width; ++i) {
          Real x2 = (i * step_f) * (i * step_f); 
          row_sum += (x2 + y2 >= 1.0f);
        }
        sum += row_sum;  
      }
    }
  }

  Real pi =  (N - static_cast<Real>(sum)) / N * 4;
  p_log << "pi:\t" << pi << '\n';
  return tuple{long(N) * 6, 0};
}
```, caption: [The $pi$ with grid calculation.])
<lst:pi-grid> 

On the other side the implementation with random numbers @lst:pi-rand, we get a avg througput of $26.6961 "GFLOPs"/s$, which is verz much slowe then the @lst:pi-grid implementation. This is expected (cant coomnet on what scale this is expcted though) since the random number generation is computationally expensive and does not vectorize well, leading to a significant performance drop compared to the grid-based approach.

#figure(
```cpp
auto reduce_pi_rand(Index N = default_N)
{
  Index num_steps = N;
  Int height = static_cast<Int>(std::sqrt(N));
  Int width = height;
  num_steps = height * width;
  Real sum;

  for (auto _ : p_loop_state)
  {
    sum = 0;

    #pragma omp parallel default(none) shared(height, width) reduction(+ : sum)
    {
      uniform_real_distribution dist{0., 1.};
      default_random_engine rng;
      rng.seed(0xAAAA ^ omp_get_thread_num()); 

      #pragma omp for schedule(static)
      for (Index j = 0; j < height; j++)
      {
        Int row_sum = 0;

        #pragma omp simd reduction(+ : row_sum)
        #pragma unroll
        for (Index i = 0; i < width; i++)
        {
          Real x1 = dist(rng);
          Real y1 = dist(rng);
          row_sum += (x1 * x1 + y1 * y1 >= 1.0f);
        }

        sum += row_sum;
      }
    }
  }
  Real pi = (N - static_cast<Real>(sum)) / N * 4;
  p_log << "pi:\t" << pi << '\n';
  return tuple{long(N) * 6, 0};
}
}
```, caption: [The $pi$ with rand calculation.])
<lst:pi-rand> 


= Saxpy and Triad

Bellow we can see the results of the triad and saxpy implementation.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("images/saxpy_bench.png", width: 80%),
    caption: [The saxpy benchmark results.]
  ),
  figure(
    image("images/triad_bench.png", width: 80%),
    caption: [The triad benchmark results.]
  ),
)<fig:saxpy_triad>

Saxpy consistently outperforms triad by 25 to 34% in all cases, which aligns with the expected bandwidth difference: triad requires reading two full arrays plus a scalar before writing, while saxpy only reads one. The gap between the two, is stable across functor, lambda and index variants, suggesting it reflects a fundamental bandwidth constraint. The extra read in triad costs roughly the same penalty regardless of how the loop is expressed.

= Bounding box calculation

For the following implementation of the bounding box calculation in @lst:bbox, we get the following results: 
`lowerleft:-1.00, -2.00 upperright: 3.00, 4.00`, which is the expected output. 


#figure(
```cpp
auto reduce_boundingbox(Index N = default_N) {
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
  {
      randindex2 = std::rand() % N;
  }

  V[std::rand() % N] = points{-1, 4};
  V[std::rand() % N] = points{3, -2};

  twoPoints res;
  
  Real temp_x_1 = numeric_limits<Real>::max();
  Real temp_y_1 = numeric_limits<Real>::max();
  Real temp_x_2 = numeric_limits<Real>::lowest();
  Real temp_y_2 = numeric_limits<Real>::lowest();

  #pragma omp parallel for reduction (min: temp_x_1, temp_y_1) reduction (max: temp_x_2, temp_y_2)
  for (size_t i = 0; i < V.size(); ++i)
  {
    const auto& point = V[i];
    temp_x_1 = std::min(temp_x_1, point.first);
    temp_y_1 = std::min(temp_y_1, point.second);
    temp_x_2 = std::max(temp_x_2, point.first);
    temp_y_2 = std::max(temp_y_2, point.second);
  }

  res.first = points{temp_x_1, temp_y_1};
  res.second = points{temp_x_2, temp_y_2};
  
  p_log << "lowerleft:\t" << temp_x_1 << "," << temp_y_1 << "\tupperright:\t" << temp_x_2 << "," << temp_y_2 << '\n';
  return tuple{N * sizeof(Real), 0};
}
```, caption: [The bounding box calculation.])
<lst:bbox> 
