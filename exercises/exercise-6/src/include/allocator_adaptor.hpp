#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <climits>
#include <limits>

namespace numa {
template <typename T, typename A = std::allocator<T>>
class default_init_allocator : public A {
  // Implementation taken from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = default_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) {
    std::allocator_traits<A>::construct(static_cast<A&>(*this), ptr,
                                        std::forward<ArgsT>(args)...);
  }
};

template <typename T, typename A = std::allocator<T>>
class no_init_allocator : public A {
  // Implementation adapted from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = no_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) { }
};

// adapted from https://stackoverflow.com/questions/60169819/modern-approach-to-making-stdvector-allocate-aligned-memory
template<typename T, std::size_t ALIGNMENT= alignof(T)>
class aligned_allocator
{
    static_assert(ALIGNMENT>=alignof(T));
public:
    using value_type= T;
    static constexpr std::align_val_t alignment{ALIGNMENT};

    template<class OtherT>
    struct rebind {
        using other= aligned_allocator<OtherT, ALIGNMENT>;
    };

    constexpr aligned_allocator() noexcept = default;
    constexpr aligned_allocator( aligned_allocator const& ) noexcept = default;

    template<typename U>
    constexpr aligned_allocator( aligned_allocator<U, ALIGNMENT> const& ) noexcept {}

    [[nodiscard]] T* allocate( std::size_t nElementsToAllocate )
	{
        if ( nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        return ::new(alignment) T[nElementsToAllocate];
    }

    void deallocate( T* allocatedPointer, [[maybe_unused]] std::size_t  nBytesAllocated )
	{
        ::delete[] allocatedPointer, alignment;
    }
};

template<typename T, size_t ALIGNMENT= alignof(T)>
struct alignas(ALIGNMENT) aligned_class : public T {
    static_assert(ALIGNMENT>=alignof(T));
    template <class... Args> aligned_class(Args&&... args) : T{std::forward<Args>(args)...} {}
};

template<typename T, size_t ALIGNMENT= alignof(T)>
struct alignas(ALIGNMENT) aligned_nonclass {
    static_assert(ALIGNMENT>=alignof(T));
    template <class... Args> aligned_nonclass(Args&&... args) : a{std::forward<Args>(args)...} {}
    operator T() const {return a;}
    T a;
};

template<typename T, size_t ALIGNMENT= alignof(T)>
using aligned_type= std::conditional<std::is_class<T>::value, aligned_class<T,ALIGNMENT>, aligned_nonclass<T,ALIGNMENT>>::type;

}  // namespace numa