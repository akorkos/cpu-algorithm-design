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
  #text(size: 1.2em, weight: "semibold")[Exercise 2]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

= Performence of reduce and transform

For reduce, the strategy you pick makes a difference. The standard STL approach 
is the slowest by a wide margin, while the vertical SIMD method comes out on top.
For transform, none of the above mentioned matter, 
since is limited by how fast data can be moved from memory, so all 
implementations end up performing nearly the same. 

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("images/ex02out_benchmarkSIMDreduce.png"), 
    caption: [Benchmarks for reduce.]
  ),
  figure(
    image("images/ex02out_benchmarkSIMDtransform.png"), 
    caption: [Benchmarks for transform.]
  ),
)

= Adapting reduce and transform

For reduce, the STL version is now by far the fastest (the opposite of the 
previous run) while all the SIMD and iterator variants cluster together closely. 
For transform, there is now more spread across implementations, the OmpSimd 
inner loop variant is the clear winner, while the plain OmpSimd iterator and 
range versions are among the slowest, and the inner loop variants in general 
show a much bigger performance gap from their non inner loop counterparts than before.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("images/ex02out_benchmarkSIMDreduceV.png"), 
    caption: [Benchmarks for reduce with view.]
  ),
  figure(
    image("images/ex02out_benchmarkSIMDtransformV.png"), 
    caption: [Benchmarks for transform with view.]
  ),
)