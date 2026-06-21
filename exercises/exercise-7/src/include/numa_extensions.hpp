#pragma once

#include <vector>
#include <utility>
#include <concepts>
#include <ranges>
#include <execution>
#include <omp.h>
// #include <oneapi/tbb/parallel_for.h>
// #include <oneapi/tbb/blocked_range.h>

#include "allocator_adaptor.hpp"
// #include "arena.hpp"

template <int>
struct tag_type
{
    // dummy type to distinguish the constructors with same signature
    // tag_type with template parameter 1 is used for constructor that takes threads_per_node
    // tag_type with template parameter 1 is used for constructor that takes an iterator
    // tag_type with template parameter 2 is used for constructor that takes granularity
    // tag_type with template parameter 3 is used for constructor that takes granularity and iterator
};


std::vector<std::pair<size_t, size_t>> compute_ranges(size_t count, int nodes, int Fold = 1)
{
    std::vector<std::pair<size_t, size_t>> node_range;
    int realSize = nodes * Fold;
    int sizePerStruct = count / Fold;
    assert(count % Fold == 0 && "Total size of container should be a multiple of Fold");

    node_range.resize(realSize);

    size_t node_size = sizePerStruct / nodes;
    size_t rest = sizePerStruct % nodes;

    size_t start = 0;
    for (int i = 0; i < nodes; i++)
    {
        size_t size = (i < rest) ? node_size + 1 : node_size;
        for (int j = 0; j < Fold; j++)
        {
            node_range[i + j * nodes] = std::make_pair(start + j * sizePerStruct, start + size + j * sizePerStruct);
        }
        start += size;
    }
    return node_range;
}

std::vector<std::pair<size_t, size_t>> compute_ranges_gran(size_t count, int nodes, int granularity, int Fold = 1)
{
    std::vector<std::pair<size_t, size_t>> node_range;
    int realSize = nodes * Fold;
    int sizePerStruct = count / Fold;
    assert(count % Fold == 0 && "Total size of container should be a multiple of Fold");

    node_range.resize(realSize);
    size_t rest = sizePerStruct;
    size_t start = 0;
    for (int i = 0; i < nodes; i++)
    {
        size_t node_size = rest / (nodes - i);
        size_t node_true_size = ((node_size + granularity - 1) / granularity) * granularity;
        size_t size = (node_true_size < rest) ? node_true_size : rest;
        for (int j = 0; j < Fold; j++)
        {
            node_range[i + j * nodes] = std::make_pair(start + j * sizePerStruct, start + size + j * sizePerStruct);
        }
        start += size;
        rest -= size;
    }
    return node_range;
}


template <typename Container>
concept container = std::random_access_iterator<typename Container::iterator> || std::contiguous_iterator<typename Container::iterator> ||
                    std::ranges::view<Container>;

template <container Container, int Fold = 1>
class numa_extensions
{
public:
    using container_type = Container;
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;

    // constructors

    numa_extensions() : container_(Container()) {}

    explicit numa_extensions(size_type count) : container_(Container(count)) {}

    //     explicit numa_extensions(size_type count, const value_type &value) : container_(Container(count))
    //     {

    // #pragma omp parallel for
    //         for (size_t i = 0; i < count; i++)
    //         {
    //             new (&container_[i]) value_type{value};
    //         }
    //     }

    //     explicit numa_extensions(numa::ArenaMgtTBB &arena, size_type count, const value_type &value) : container_(Container(count))
    //     {
    //         node_ranges(count, arena.get_nodes());
    //         arena.execute([&](const int i)
    //                       { tbb::parallel_for(tbb::blocked_range<size_t>(node_range[i].first, node_range[i].second), [&](const tbb::blocked_range<size_t> r)
    //                                           {
    // #pragma omp simd
    //                 for (auto i = r.begin(); i < r.end(); i++) {
    //                     new(&container_[i]) value_type{value};
    //                 } }, tbb::static_partitioner()); });
    //     }
    explicit numa_extensions(size_type count, const value_type &value) : container_(Container(count))
    {
        const int nodes = omp_get_num_places();

        auto node_range = compute_ranges(count, nodes);
#pragma omp parallel proc_bind(spread) num_threads(nodes)
        {
            auto node = omp_get_place_num();
            for (int i = 0; i < Fold; i++)
            {
                int idx = node + nodes * i;
                std::uninitialized_fill(std::execution::unseq, container_.begin() + node_range[idx].first, container_.begin() + node_range[idx].second, value);
            }
        }
    }
    explicit numa_extensions(size_type count, const value_type &value, int granularity) : container_(Container(count))
    {
        const int nodes = omp_get_num_places();
        auto node_range = compute_ranges(count, nodes, granularity);
#pragma omp parallel proc_bind(spread) num_threads(nodes)
        {
            auto node = omp_get_place_num();
            for (int i = 0; i < Fold; i++)
            {
                int idx = node + nodes * i;
                std::uninitialized_fill(std::execution::unseq, container_.begin() + node_range[idx].first, container_.begin() + node_range[idx].second, value);
            }
        }
    }

    template <std::random_access_iterator ForwardIt>
    explicit numa_extensions(ForwardIt first, ForwardIt second) : container_(Container(second - first))
    {
        const int nodes = omp_get_num_places();
        auto count = second - first;
        auto node_range = compute_ranges(count, nodes);
#pragma omp parallel proc_bind(spread) num_threads(nodes)
        {
            auto node = omp_get_place_num();
            for (int i = 0; i < Fold; i++)
            {
                int idx = node + nodes * i;
                std::uninitialized_copy(std::execution::unseq, first + node_range[idx].first, first + node_range[idx].second, container_.begin() + node_range[idx].first);
            }
        }
    }
    template <std::random_access_iterator ForwardIt>
    explicit numa_extensions(ForwardIt first, ForwardIt second, int granularity) : container_(Container(second - first))
    {
        // node_ranges(count, nodes);
        const int nodes = omp_get_num_places();
        auto count = second - first;
        auto node_range = compute_ranges(count, nodes, granularity);
// #pragma omp parallel proc_bind(spread) num_threads(nodes)
#pragma omp parallel proc_bind(spread) num_threads(nodes)
        {
            auto node = omp_get_place_num();
            for (int i = 0; i < Fold; i++)
            {
                int idx = node + nodes * i;
                std::uninitialized_copy(std::execution::unseq, first + node_range[idx].first, first + node_range[idx].second, container_.begin() + node_range[idx].first);
            }
        }
    }

    //     explicit numa_extensions(int thrds_per_node, size_type count, const value_type &value, tag_type<1> dummy) : container_(Container(count))
    //     {
    //         const int nodes = omp_get_num_places();
    //         node_ranges(count, nodes);
    // #pragma omp parallel proc_bind(spread) num_threads(nodes)
    //         {
    //             const auto range = node_range[omp_get_place_num()];
    // #pragma omp parallel for proc_bind(master) num_threads(thrds_per_node)
    //             for (size_t i = range.first; i < range.second; i++)
    //             {
    //                 new (&container_[i]) value_type{value};
    //             }
    //         }
    //     }

    numa_extensions(numa_extensions<Container, Fold> &&obj) : container_(std::move(obj.container_))
    {
    }

    void scan_init(size_type block_size, const value_type &init)
    {
        int threads = omp_get_num_procs();
        if (block_size > (container_.size() / threads))
            block_size = container_.size() / threads;
        size_t iterations = container_.size() / (block_size * threads);
        size_t rest = container_.size() % (block_size * threads);
        int rest_threads = rest / block_size;
        int cores = threads / 2;

// #pragma omp parallel proc_bind(spread) num_threads(threads) firstprivate(iterations)
#pragma omp parallel proc_bind(spread) num_threads(threads) firstprivate(iterations)
        {
            int tid = omp_get_thread_num();
            int block_index = (tid % 2) ? ((tid - 1) / 2 + cores) : (tid / 2);
            if (rest && block_index < rest_threads)
                iterations++;
            for (int i = 0; i < iterations; i++)
            {
                size_t start = i * threads * block_size + block_index * block_size;
                std::uninitialized_fill_n(&container_[start], block_size, init);
            }
        }
    }

    ~numa_extensions() = default;

    constexpr size_type size() const noexcept { return container_.size(); }

    constexpr value_type *data() noexcept { return container_.data(); }

    constexpr iterator begin() noexcept { return container_.begin(); }

    constexpr iterator end() noexcept { return container_.end(); }

    constexpr const_iterator cbegin() const noexcept { return container_.cbegin(); }

    constexpr const_iterator begin() const { return container_.cbegin(); }

    constexpr const_iterator cend() const noexcept { return container_.cend(); }

    constexpr const_iterator end() const { return container_.cend(); }

    reference operator[](size_type index) { return container_[index]; }

    reference back() { return container_.back(); };

    // std::pair<size_t, size_t> get_range(size_type index) { return node_range[index]; }
    // std::pair<size_t, size_t> get_Stride_range(size_type index, size_type start, size_type end, size_type stride)
    // {
    //     auto range = get_overlap_range(index, start, end);
    //     auto first = range.first >= start ? range.first - start : 0;
    //     // auto second = range.second >= end ? end: range.second;
    //     return std::pair<size_t, size_t>{(first + stride - 1) / stride, (range.second - start + stride - 1) / stride};
    // }

    // std::pair<size_t, size_t> get_overlap_range(size_type index, size_type start, size_type end)
    // {
    //     auto range = node_range[index];
    //     size_t rangefirst = std::max(range.first, start);
    //     size_t rangesecond = std::min(range.second, end);

    //     if (rangefirst >= rangesecond)
    //     {
    //         return {0, 0};
    //     }
    //     return std::pair<size_t, size_t>{rangefirst, rangesecond};
    // }

    const Container &get_container() { return container_; }
    Container container_;

private:
    // void node_ranges(size_type count, int nodes)
    // {
    //     compute_ranges(count, nodes, Fold);
    // }
    // void node_ranges(size_type count, int nodes, int granularity)
    // {
    //     // reserve vs resize
    //     int realSize = nodes * Fold;
    //     int sizePerStruct = count / Fold;
    //     assert(count % Fold == 0 && "Total size of container should be a multiple of Fold");

    //     node_range.resize(realSize);
    //     size_t rest = sizePerStruct;
    //     size_t start = 0;
    //     for (int i = 0; i < nodes; i++)
    //     {
    //         size_t node_size = rest / (nodes - i);
    //         size_t node_true_size = ((node_size + granularity - 1) / granularity) * granularity;
    //         size_t size = (node_true_size < rest) ? node_true_size : rest;
    //         for (int j = 0; j < Fold; j++)
    //         {
    //             node_range[i + j * nodes] = std::make_pair(start + j * sizePerStruct, start + size + j * sizePerStruct);
    //         }
    //         start += size;
    //         rest -= size;
    //     }
    // }

    // std::vector<std::pair<size_t, size_t>> node_range;
};


std::pair<size_t, size_t> get_overlap_range(std::vector<std::pair<size_t, size_t>> const &node_range, size_t index, size_t start, size_t end)
{
    auto range = node_range[index];
    size_t rangefirst = std::max(range.first, start);
    size_t rangesecond = std::min(range.second, end);

    if (rangefirst >= rangesecond)
    {
        return {0, 0};
    }
    return std::pair<size_t, size_t>{rangefirst, rangesecond};
}

std::pair<size_t, size_t> get_Stride_range(std::vector<std::pair<size_t, size_t>> const &node_range, size_t index, size_t start, size_t end, size_t stride)
{
    auto range = get_overlap_range(node_range, index, start, end);
    auto first = range.first >= start ? range.first - start : 0;
    // auto second = range.second >= end ? end: range.second;
    return std::pair<size_t, size_t>{(first + stride - 1) / stride, (range.second - start + stride - 1) / stride};
}

