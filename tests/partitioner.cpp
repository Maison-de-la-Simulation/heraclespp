// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <compare>
#include <iterator>
#include <set>
#include <type_traits>
#include <version>

#include <gtest/gtest.h>

#include <Kokkos_Array.hpp>
#include <range.hpp>

class Partitioner
{
private:
    std::array<int, 2> m_dom;

    int m_block;

public:
    class Iterator;

    Partitioner(int size, int block) noexcept;

    [[nodiscard]] auto begin() const noexcept -> Iterator;

    [[nodiscard]] auto end() const noexcept -> Iterator;
};

class Partitioner::Iterator
{
private:
    int m_front;

    int m_end;

    int m_block;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::array<int, 2>;
    using difference_type = std::make_signed_t<int>;

    Iterator() = default;

    explicit Iterator(int const begin, int const end, int const block) noexcept : m_front(begin), m_end(end), m_block(std::min(block, end - begin)) {}

    auto operator*() const noexcept -> std::array<int, 2>
    {
        return std::array<int, 2> {m_front, m_front + m_block};
    }

    auto operator++() noexcept -> Iterator&
    {
        m_front += m_block;
        if (m_front + m_block > m_end) {
            m_block = m_end - m_front;
        }
        return *this;
    }

    friend auto operator==(Iterator const& lhs, Iterator const& rhs) -> bool
    {
        return lhs.m_front == rhs.m_front;
    }

#if !defined(__cpp_lib_three_way_comparison)
    friend bool operator!=(Iterator const& lhs, Iterator const& rhs)
    {
        return lhs.m_front != rhs.m_front;
    }
#endif

#if defined(__cpp_lib_three_way_comparison)
    friend auto operator<=>(Iterator const& lhs, Iterator const& rhs) -> std::strong_ordering
    {
        return lhs.m_front <=> rhs.m_front;
    }
#else
    friend bool operator>(Iterator const& lhs, Iterator const& rhs)
    {
        return lhs.m_front > rhs.m_front;
    }

    friend bool operator<(Iterator const& lhs, Iterator const& rhs)
    {
        return lhs.m_front < rhs.m_front;
    }

    friend bool operator>=(Iterator const& lhs, Iterator const& rhs)
    {
        return lhs.m_front >= rhs.m_front;
    }

    friend bool operator<=(Iterator const& lhs, Iterator const& rhs)
    {
        return lhs.m_front <= rhs.m_front;
    }
#endif
};

Partitioner::Partitioner(int const size, int const block) noexcept : m_dom {0, size}, m_block(block) {}

auto Partitioner::begin() const noexcept -> Partitioner::Iterator
{
    return Iterator(m_dom[0], m_dom[1], m_block);
}

auto Partitioner::end() const noexcept -> Partitioner::Iterator
{
    return Iterator(m_dom[1], m_dom[1], m_block);
}

TEST(Partitionner, SomeTest)
{
    std::array<int, 3> const nx {9, 7, 3};
    int const nghosts = 2;
    std::array<int, 3> const block {3, 3, 3};
    std::set<std::array<int, 3>> list_of_cells;

    {
        hclpp::Range const rng({0, 0, 0}, nx, nghosts);
        auto const [begin, end] = hclpp::cell_range(rng.no_ghosts());
        auto const [ibegin, jbegin, kbegin] = begin;
        auto const [iend, jend, kend] = end;
        for (int k = kbegin; k < kend; ++k) {
            for (int j = jbegin; j < jend; ++j) {
                for (int i = ibegin; i < iend; ++i) {
                    std::array<int, 3> const elem {i, j, k};
                    list_of_cells.emplace(elem);
                }
            }
        }
    }
    EXPECT_FALSE(list_of_cells.empty());

    for (std::array<int, 2> const rng2 : Partitioner(nx[2], block[2])) {
        for (std::array<int, 2> const rng1 : Partitioner(nx[1], block[1])) {
            for (std::array<int, 2> const rng0 : Partitioner(nx[0], block[0])) {
                hclpp::Range const rng(rng0, rng1, rng2, nghosts);
                auto const [begin, end] = hclpp::cell_range(rng.no_ghosts());
                auto const [ibegin, jbegin, kbegin] = begin;
                auto const [iend, jend, kend] = end;
                for (int k = kbegin; k < kend; ++k) {
                    for (int j = jbegin; j < jend; ++j) {
                        for (int i = ibegin; i < iend; ++i) {
                            std::array<int, 3> const elem {rng0[0] + i, rng1[0] + j, rng2[0] + k};
                            list_of_cells.erase(elem);
                        }
                    }
                }
            }
        }
    }

    EXPECT_TRUE(list_of_cells.empty());
}
