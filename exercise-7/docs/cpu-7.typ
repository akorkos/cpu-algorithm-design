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
  #text(size: 1.2em, weight: "semibold")[Exercise 7]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

= Remark

Sorry for the late submission, I wanted to test the code yesterday but all the nodes were for the most part used.

#set heading(numbering: "1.")
#set heading(outlined: true)

= Loop Dependency

The results for all the implemations can be seen here: 

#figure(
  image("images/Dependency.png", width: 80%),
  caption: [Loop dependencies for the different implementations]
)<fig:res>

The `update_parallel_simd` version delivers 25.7 GB/ with good stability (CV=0.00067), probably because SIMD vectorization within stride chunks avoids thread overhead entirely. The `update_parallel_nested_loop` is about 1.27 GB/s which is much slower than sequential. In such, the parallel region overhead is only paid back when the work is sufficiently large, and the inner loop parallelization of small fixed size chunks is almost always counterproductive.

= Another loop dependency
Based on @fig:res, the `between_iter_twoloops` (56.8 GB/s) out performs `between_iter_optimal` (15.6 GB/s) by breaking the V to W dependency into two separate parallel loops, trading an extra memory pass for full parallel throughput. The `between_iter_optimal` fuses both computations sequentially to minimize memory traffic but remains strictly serial due to the shifted dependency of `W[i+1] = func2(V[i])`, bottlenecking the throughput. On large N, parallelism wins over cache-friendly fusion.

For the `between_iter_task`, i tried to model it after the fibonacci example from #link("https://hpc2n.github.io/Task-based-parallelism/branch/master/task-basics-2/", "here") but always led to infinte loops.