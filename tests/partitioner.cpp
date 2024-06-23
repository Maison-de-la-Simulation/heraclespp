#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <range.hpp>

class Partitioner
{
private:
    std::array<int, 2> m_dom;

    int m_block;

public:
    class iterator;

    Partitioner(int size, int block) noexcept;

    iterator begin() const noexcept;

    iterator end() const noexcept;
};

class Partitioner::iterator
{
private:
    int m_front;

    int m_end;

    int m_block;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::array<int, 2>;
    using difference_type = std::make_signed_t<int>;

    iterator() = default;

    explicit iterator(int const begin, int const end, int const block) noexcept
        : m_front(begin)
        , m_end(end)
        , m_block(std::min(block, end - begin))
    {
    }

    std::array<int, 2> operator*() const noexcept
    {
        return std::array<int, 2> {m_front, m_front + m_block};
    }

    iterator& operator++() noexcept
    {
        m_front += m_block;
        if (m_front + m_block > m_end)
        {
            m_block = m_end - m_front;
        }
        return *this;
    }

    friend bool operator==(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front == rhs.m_front;
    }

#ifndef __cpp_lib_three_way_comparison
    friend bool operator!=(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front != rhs.m_front;
    }
#endif

#ifdef __cpp_lib_three_way_comparison
    friend std::strong_ordering operator<=>(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front <=> rhs.m_front;
    }
#else
    friend bool operator>(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front > rhs.m_front;
    }

    friend bool operator<(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front < rhs.m_front;
    }

    friend bool operator>=(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front >= rhs.m_front;
    }

    friend bool operator<=(iterator const& lhs, iterator const& rhs)
    {
        return lhs.m_front <= rhs.m_front;
    }
#endif
};

Partitioner::Partitioner(int const size, int const block) noexcept : m_dom {0, size}, m_block(block)
{
}

Partitioner::iterator Partitioner::begin() const noexcept
{
    return iterator(m_dom[0], m_dom[1], m_block);
}

Partitioner::iterator Partitioner::end() const noexcept
{
    return iterator(m_dom[1], m_dom[1], m_block);
}

TEST(Partitionner, SomeTest)
{
    std::array<int, 3> const nx {9, 7, 3};
    int const nghosts = 2;
    std::array<int, 3> const block {3, 3, 3};
    std::set<std::array<int, 3>> list_of_cells;

    {
        novapp::Range const rng({0, 0, 0}, nx, nghosts);
        auto const [begin, end] = novapp::cell_range(rng.no_ghosts());
        auto const [ibegin, jbegin, kbegin] = begin;
        auto const [iend, jend, kend] = end;
        for (int k = kbegin; k < kend; ++k)
        {
            for (int j = jbegin; j < jend; ++j)
            {
                for (int i = ibegin; i < iend; ++i)
                {
                    std::array<int, 3> const elem {i, j, k};
                    list_of_cells.emplace(elem);
                }
            }
        }
    }
    EXPECT_FALSE(list_of_cells.empty());

    for (std::array<int, 2> const rng_z : Partitioner(nx[2], block[2]))
    {
        for (std::array<int, 2> const rng_y : Partitioner(nx[1], block[1]))
        {
            for (std::array<int, 2> const rng_x : Partitioner(nx[0], block[0]))
            {
                novapp::Range const rng(rng_x, rng_y, rng_z, nghosts);
                auto const [begin, end] = novapp::cell_range(rng.no_ghosts());
                auto const [ibegin, jbegin, kbegin] = begin;
                auto const [iend, jend, kend] = end;
                for (int k = kbegin; k < kend; ++k)
                {
                    for (int j = jbegin; j < jend; ++j)
                    {
                        for (int i = ibegin; i < iend; ++i)
                        {
                            std::array<int, 3> const elem {rng_x[0] + i, rng_y[0] + j, rng_z[0] + k};
                            list_of_cells.erase(elem);
                        }
                    }
                }
            }
        }
    }

    EXPECT_TRUE(list_of_cells.empty());
}
