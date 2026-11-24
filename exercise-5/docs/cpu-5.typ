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
  #text(size: 1.2em, weight: "semibold")[Exercise 5]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

= Saxpy vs Triad

1. The number of reads, updates, and writes are summed up below:

#figure(
  table(
    columns: 4,
    stroke: none,
    [], [*Reads*], [*Writes*], [*Updates*],
    [Saxpy:], [2N], [-], [N],
    [Triad:], [2N], [N], [-],
  )
)

The difference between the two is that in Triad we do not update since we save the 
result in another vector; in comparison, in Saxpy, the result is saved in the 
same position of the vector $W$.

2. Both Saxpy and Triad need $2N$ FLOPs. For operations using float we get $12N$
bytes moved for Saxpy and for Triad. Hence, we calculate the arithmetic 
intensity as follows:

$
  text("for Saxpy & Triad:") (2N) / (12N) = 1/6
$

For double operations we get: 

$
  text("for Saxpy & Triad:") (2N) / (24N) = 1/12
$

= Average bandwidth

1. The amount of data sent is: 
$
  D = (B_0 * T_0) + (B_1 * T_1) + (B_2 * T_2) 
$

and the total time is: $T = T_0 + T_1 + T_2$.

Thus: 
$
  B = D / T = ((B_0 * T_0) + (B_1 * T_1) + (B_2 * T_2)) / (T_0 + T_1 + T_2).
$ <eq:4>

2. Using $T_i = a_i * T, i in [0,2]$ in @eq:4, we remove the time $T$ dependency.
So we get:

$
  B =((B_0 * a_0) + (B_1 * a_1) + (B_2 * a_2)) / (a_0 + a_1 + a_2).
$

3. We solve @eq:4 for T: 

$
  T = D / B.
$

Here $D$ would be the sum of the bytes transferred and $B$ the sum of bytes sent 
at a certain bandwidth, i.e.: 

$
  dots => T = (N_0 + N_1 + N_2) / (N_0/B_0 + N_1/B_1 + N_2/B_2).
$

4. Again, assuming $N_i = a_i * N, i in [0,2]$, the $N$ factors out, thus:

$
  T = (a_0 + a_1 + a_2) / (a_0/B_0 + a_1/B_1 + a_2/B_2).
$
= Constant stencil with LLC cache

For a $27$-point stencil we have: load $1$ float from DRAM, load $27$ floats from LLC, 
store $1$ float to DRAM. 

$
  "Latency: " 110 + 4 / 200 + 10 + (27*4)/2000 + (27*5) + 4 / 200 + 110 approx 365 "ns". 
$

$
  "Throughput: " 200/(4+4) = 25 "Gupdates"/s.
$

$
  "DRAM bandwidth: " 200 "GB"/s.
$

$
  "LLC bandwidth: " 27*100 = 2700 "GB"/s.
$

We can assume that the LLC is large enough to hold all the neighbors of the 27-point stencil. The LLC 
bandwidth in the original was $2000 "GB"/s$, but in our case, we need $2700 "GB"/s$, so we need to enlarge the LLC bandwidth. 

= Performance in real life

Examining @lst:stencil we can see the following: in the 2nd implementation, the 
inner looping leads to sequentially accessing the vectors, i.e., for `W[y*Nx+x]` and 
`V[(y+sy)*NX+(x+sx)]`. Additionally, the `S[sy*Sx+sx]` value is a constant for the inner looping, 
so it does not need to be read at each iteration.

In the 1st implementation, the only sequential accessing is for `S[sy*Sx+sx]`, but 
for `V[(y+sy)*NX+(x+sx)]` we "jump" around the memory to access the value of the 
vector. The `W[y*Nx+x]` for the inner looping always stays the same, i.e., we 
access the same location $"Srx" * "Sry"$ times. For the `S` vector, the elements
are accessed sequentially in the inner looping.

Summing up, the 2nd implementation is more efficient.

#figure(
```cpp
// implementation 1
auto Sx= Srx-Slx+1;
auto Sy= Sry-Sly+1;
auto NX= Nx+Sx-1;
auto NY= Ny+Sy-1;
Container V_(NX*NY, init.begin(), init.end());
auto V= V_.begin()-(Sly*NX+Slx);
Container W(Nx*Ny, 0);

for(Index y= 0; y<Ny; ++y) {
  for(Index x= 0; x<Nx; ++x) {
    for(Index sy= Slx; sy<=Srx; ++sy) {
      for(Index sx= Sly; sx<=Sry; ++sx) {
        W[y*Nx+x]+= S[sy*Sx+sx]*V[(y+sy)*NX+(x+sx)];
      }
    }
  }
}

//implementation 2
auto Sx= Srx-Slx+1;
auto Sy= Sry-Sly+1;
auto NX= Nx+Sx-1;
auto NY= Ny+Sy-1;
Container V_(NX*NY, init.begin(), init.end());
auto V= V_.begin()-(Sly*NX+Slx);
Container W(Nx*Ny, 0);

for(Index sy= Slx; sy<=Srx; ++sy) {
  for(Index sx= Sly; sx<=Sry; ++sx) {
    for(Index y= 0; y<Ny; ++y) {
      for(Index x= 0; x<Nx; ++x) {
        W[y*Nx+x]+= S[sy*Sx+sx]*V[(y+sy)*NX+(x+sx)];
      }
    }
  }
}
```, caption: [Stencil looping.])
<lst:stencil> 
