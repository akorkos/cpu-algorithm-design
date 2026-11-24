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
  #text(size: 1.2em, weight: "semibold")[Exercise 9]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

= for_each Subrange

+ The `if` is replaced by a ternary
  `W[i] = (i >= 2 && i < N-3) ? a*V[i] : b*V[i];` with `b = 4`. The comparison
  compiles to a SIMD mask/blend, so all lanes stay vectorized under `omp for simd`
  with no branch. It reaches *86.3 GB/s*.

+ `Q(N-5)` holds the integers `[2,N-3)`; each iteration rebuilds
  `indices` in parallel via `indices[k]=k+2` (no `push_back`), then gathers
  `W[indices[i]]=a*V[indices[i]]`. Throughput drops to *55.3 GB/s*, i.e. about
  *x1.56 slower* than `Subrange_interval`, caused by the extra index array and
  the indirect (gather) access instead of a contiguous stream.

+ `Q(N-5)` stores unit intervals as pairs `[2,3),[3,4),…`; `intervals`
  is filled in parallel (`intervals[k]={k+2,k+3}`) and the compute uses `.first`:
  `W[p.first]=a*V[p.first]`. This is by far the slowest at *7.49 GB/s* (~x11.5 slower
  than interval): the `pair` doubles index traffic and the strided `.first` access
  breaks the contiguous SIMD loads.

+ Yes, all three run in one fused `omp for simd` over
  `[0,Nrow)`: `W[vStride0[i]]=V`, `W[vStride1[i]]=a*V`, `W[vStride2[i]]=a*V*V`.
  The three index sets are disjoint, so there are no write conflicts; the single
  pass reaches *31.8 GB/s*.

= for_each Reordering

#[
#set enum(start: 5)
+ With `default_N = 1<<Nexp`, index `i` is mapped by
  reversing its lower `Nexp` bits and gathering `V[rev(i)]`. It runs at only
  *11.1 GB/s*, much slower than a plain sequential reverse, because the access
  pattern is scattered and cache-unfriendly. `N` must be `2^Nexp` so the reversed
  bits stay within `[0,N)` and the map is a bijection; otherwise it is not a permutation.

+ A second rotation `P1[i]=(i+m)%N` is added, then the two are
  fused once into `Q[i]=P1[P[i]]`, so the hot loop `W[i]=V[Q[i]]` equals applying `P`
  then `P1` (a single rotation by `n+m`). Precomputing `Q` keeps the loop one
  contiguous-source gather at *102 GB/s* (fastest kernel overall).

+ The store is guarded so only one of the `n` colliding
  indices writes to `W[i/n]`: `if(i%n==n-1) W[i/n]=V[i];`. This removes the write
  race deterministically (here the last competing index wins, since we use `i%n==0` for the
  first). It runs at *80.8 GB/s*.

+ `Reordering_is_permutation` sorts `V` in parallel, then counts
  mismatches with a reduction: `parallel for reduction(+:ans)` over
  `ans += (V[i]!=i)`; `ans==0` ⇒ `V` is a bijection `[0,N)→[0,N)`. The run prints
  `success` at *23.6 GB/s*.
]

#figure(
  image("images/omp_foreach_w08.png", width: 100%),
  caption: [Average kernel time (ns).],
)