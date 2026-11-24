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
  #text(size: 1.2em, weight: "semibold")[Exercise 3]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

The 1. exercise form adapting reduce and transform and adapting `benchReduceUnrollTreeDirective` were developed.

The figure below shows that SIMD and unrolling methods give relatively low performance, while more complex approaches greatly improve the benchmarks. Manual unrolling and directive unrolling provide only small gains, but loop peeling and tree-style reductions produce dramatic speedups by reducing reduction dependencies and improving optimization opportunities. The `benchReduceUnrollTreeDirective` performs best overall, suggesting that combining hierarchical reduction with compiler unrolling is highly effective. The `benchReduceUnrollSimdXHorizontal` and `benchReduceUnrollSimdXVertical` were not implemented, so their values are not meaningful for comparison.

#figure(
  image("images/ex03.png", width: 60%), 
  caption: [Benchmarks for reduce.]
)


