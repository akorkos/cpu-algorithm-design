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
  #text(size: 1.2em, weight: "semibold")[Exercise 1]
  #v(0.6em)
  #datetime.today().display("[day padding:none] [month repr:long] [year]")
]

#set heading(numbering: "1.")
#set heading(outlined: true)

= Basic Modern C++ STL Class Templates Theory

1. The class template `std::optional` is used when a value may or may not be present. It can be returned from a function when the function may not produce a result. In this way, it indicates that no value was produced.

2. `std::variant` works like this: we define a variant with some data types. Each object of this variant can have only one of the data types defined by the variant. A `std::tuple`, on the other hand, is a data structure where each object of this tuple contains all of the specified types.

3. `std::complex` is used to represent complex numbers in C++. Using `std::complex`, a number of mathematical operations can be performed. On the other side, `std::pair` is used for storing just a pair of values together. With that being said, one could also use `std::pair` for complex numbers, but without the methods already provided by `std::complex`. \ Both `std::tuple` and `std::array` are data structures. `std::array` can only store items of the same data type. `std::tuple`, on the other hand, can store items of different data types.

4. `std::pair` is a special case of `std::tuple`. The former holds only two items, while the latter can store one or more items. \ The difference between `std::array` and `std::complex` is that the array is a data structure, while `std::complex` is a specialized class for the representation of complex numbers.

5. `std::complex` is a specialized mathematical template that cannot be generalized. It is used for complex mathematical operations.

= Basic Modern C++ STL Class Templates Practice

== Accessors

```cpp
double get_first_double(Foo const &foo) {
    Bar bar;

    if (foo && std::holds_alternative<Bar>(*foo)) {
        bar = std::get<Bar>(*foo);

        auto array = std::get<0>(bar);

        auto pair = array[0];

        auto item_0 = std::get<0>(pair);

        return item_0.real();
    } else {
        return 42.;
    }
}

int get_first_int(Foo const &foo) {
    Bar bar;

    if (foo && std::holds_alternative<Bar>(*foo)) {
        bar = std::get<Bar>(*foo);

        return std::get<0>(bar).front().second;
    } else {
        return 42;
    }
}

float get_first_float(Foo const &foo) {
    Bar bar;

    if (foo && std::holds_alternative<Bar>(*foo)) {
        bar = std::get<Bar>(*foo);

        auto& [array, pair] = bar;

        return pair.first.front().real();
    } else {
        return 42.f;
    }
}

bool get_first_bool(Foo const &foo) {
    Bar bar;

    if (foo && std::holds_alternative<Bar>(*foo)) {
        bar = std::get<Bar>(*foo);

        auto& [array, pair] = bar;

        return pair.second;
    } else {
        return false;
    }
}```

== How Does This `std::function`?

```cpp
Bar get_Bar(Foo const &foo) {
    if (!foo) {
        return Bar();
    } else if (std::holds_alternative<Bar>(*foo)) {
        return std::get<0>(*foo);
    }

    auto f = std::get<std::function<Bar(bool)>>(*foo);
    return f(true);
}
```