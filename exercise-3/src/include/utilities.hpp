#include <algorithm>
#include <functional> // for std::plus and std::multiplies
#include <ranges>
#include <execution>

#include <vector>
#include <deque>
#include <complex>

#include <iostream>
#include <iomanip>

#include <cassert>
#include <chrono>
#pragma once

//---------------------------------------------------------------------------------------------
// move this to separate headers
//---------------------------------------------------------------------------------------------

// record time
class ClockRecorder : std::vector<std::chrono::time_point<std::chrono::system_clock>> {
public:
  using BaseVec = std::vector<std::chrono::time_point<std::chrono::system_clock>>;
  ClockRecorder(const BaseVec& vec) : BaseVec(vec) {}
  using BaseVec::BaseVec;

  struct iterator : public BaseVec::iterator {
    using BaseIt = BaseVec::iterator;
    iterator(const BaseIt& it) : BaseIt(it) {}
    using BaseIt::BaseIt;

    using BaseIt::operator++;
    iterator& operator++() {
      *(*this) = std::chrono::system_clock::now();
      BaseIt::operator++();
      return *this;
    }
  };

  iterator begin() { return iterator(BaseVec::begin()); }
  iterator end()   { return iterator(BaseVec::end()); }
};

// helper type for std::visit
template<typename... T_s>
struct zip_lambdas : T_s... 
{
  using T_s::operator()...;
};

// sign func with output -1,0,+1 or output -1,+1
template<typename T> requires std::is_arithmetic_v<T> auto sgn0(T t) { return (t>T{0})  - (t<T{0}); }
template<typename T> requires std::is_arithmetic_v<T> auto sgn1(T t) { return (t>=T{0}) - (t<T{0}); }

// div helper for integral types
template<std::integral T, std::integral U> auto div_floor(T t, U u) { return std::div(t,u); }
template<std::integral T, std::integral U> auto div_ceil(T t, U u) {
  auto r = std::div(t+sgn0(t)*(abs(u)-1),u);
  r.rem= sgn0(r.rem) * (abs(u)-abs(r.rem));
  return r;
}

//---------------------------------------------------------------------------------------------

// iterator for sequence [b, b+1,...)
template<std::integral T>
auto itSeq(T b) {
  return std::cbegin(std::views::iota(b));
}

// view of sequence [b,b+s,...,e)
template<std::integral T>
auto vSeq(T b, T e, T s=1) {
  assert(b<=e && s>=1);
  return std::views::stride(std::views::iota(b,e), s);
}

// view of sequence [b,b+s,...,b+n)
template<std::integral T>
auto vSeq_n(T b, T n, T s=1) {
  assert(n>=0 && s>=1);
  return vSeq(b, b+n, s);
}

// use an integral view WITH data from obj, some random access object - todo make a proper adaptor
template<typename T>
requires std::ranges::random_access_range<std::decay_t<T>> || std::random_access_iterator<std::decay_t<T>>
auto vWith(T&& obj) {
  return std::views::transform([&obj](std::integral auto i) -> std::decay_t<T>::reference {
    return obj[i];
  });
}

//---------------------------------------------------------------------------------------------

// stream operator<< for std::pair
//template <typename T0, typename T1>
//std::ostream& operator<<(std::ostream& os, const std::pair<T0,T1>& two) {
//  return os<<std::setw(1)<<'('<<std::setw(2)<<std::get<0>(two)<<','<<std::setw(2)<<std::get<1>(two)<<')';
//}

// stream operator<< for std::variant
template <typename... T_s>
std::ostream& operator<<(std::ostream& os, const std::variant<T_s...>& var) {
  std::visit([&os,ind0=var.index()](auto&& val) { os<<std::setw(1)<<'('<<ind0<<':'<<std::setw(2)<<val<<')'; }, var);
  return os;
}

// stream operator<< for tuple-like objects
template <template<typename...> typename TT, typename... T_s>
requires requires (TT<T_s...> t) { std::get<0>(t); } // todo use tuple-like concept
std::ostream& operator<<(std::ostream& os, const TT<T_s...>& tup) {
  char sep = '(';
  std::apply([&os,&sep](auto&&... val_s) { (..., (os<<std::setw(1)<<sep<<std::setw(2)<<val_s, sep=',')); }, tup);
  return os<<((sep=='(')?"()":")");
}

// stream operator<< for iterable non-chararcter objects (todo replace <char> with <U>)
template <typename T>
requires std::ranges::range<T> && (!std::convertible_to<T,std::basic_string_view<char>>)
std::ostream& operator<<(std::ostream& os, const T& obj) {
  char sep = '(';
  for(auto&& val : obj) { (os<<std::setw(1)<<sep<<std::setw(2)<<val, sep=','); };
  return os<<((sep=='(')?"()":")");
}

//---------------------------------------------------------------------------------------------
