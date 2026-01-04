// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace hclpp {

enum class access_mode : unsigned char { read_only, read_write, discard_write };

enum class access_target : unsigned char { device, host };

template <access_target AT>
struct AccessTargetTag
{
    static constexpr access_target value = AT;
};

template <access_mode AM>
struct AccessModeTag
{
    static constexpr access_mode value = AM;
};

template <access_target AT, access_mode AM>
struct AccessTag
{
};

inline constexpr AccessTargetTag<access_target::device> device;
inline constexpr AccessTargetTag<access_target::host> host;

inline constexpr AccessModeTag<access_mode::read_only> read_only;
inline constexpr AccessModeTag<access_mode::read_write> read_write;
inline constexpr AccessModeTag<access_mode::discard_write> discard_write;

inline constexpr AccessTag<access_target::device, access_mode::read_only> device_read_only;
inline constexpr AccessTag<access_target::device, access_mode::read_write> device_read_write;
inline constexpr AccessTag<access_target::device, access_mode::discard_write> device_discard_write;
inline constexpr AccessTag<access_target::host, access_mode::read_only> host_read_only;
inline constexpr AccessTag<access_target::host, access_mode::read_write> host_read_write;
inline constexpr AccessTag<access_target::host, access_mode::discard_write> host_discard_write;

template <access_mode AM>
constexpr auto has_read_access(AccessModeTag<AM> const mode) noexcept -> bool
{
    return mode.value == access_mode::read_only || mode.value == access_mode::read_write;
}

template <access_mode AM>
constexpr auto has_write_access(AccessModeTag<AM> const mode) noexcept -> bool
{
    return mode.value == access_mode::read_write || mode.value == access_mode::discard_write;
}

template <access_target AT, access_mode AM>
constexpr auto get_target(AccessTag<AT, AM> /*access*/) noexcept -> AccessTargetTag<AT>
{
    return AccessTargetTag<AT>();
}

template <access_target AT, access_mode AM>
constexpr auto get_mode(AccessTag<AT, AM> /*access*/) noexcept -> AccessModeTag<AM>
{
    return AccessModeTag<AM>();
}

template <typename DataType, typename Layout = Kokkos::LayoutRight, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class DualView
{
    template <typename DT, typename L, typename MS>
    friend class DualView;

    using traits = Kokkos::ViewTraits<DataType, Layout, MemorySpace, Kokkos::MemoryTraits<>>;

    using non_const_view_device_type = Kokkos::
            View<typename traits::non_const_data_type, typename traits::array_layout, typename traits::memory_space, typename traits::memory_traits>;

    using non_const_view_host_type = Kokkos::View<
            typename traits::non_const_data_type,
            typename traits::array_layout,
            typename traits::host_mirror_space::memory_space,
            typename traits::memory_traits>;

    using modified_flag_type = Kokkos::View<unsigned int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<>>;

public:
    using view_device_type
            = Kokkos::View<typename traits::data_type, typename traits::array_layout, typename traits::memory_space, typename traits::memory_traits>;

    using view_host_type = Kokkos::View<
            typename traits::data_type,
            typename traits::array_layout,
            typename traits::host_mirror_space::memory_space,
            typename traits::memory_traits>;

    template <access_mode AM, access_target AT>
    using view_type = Kokkos::View<
            std::conditional_t<AM == access_mode::read_only, typename traits::const_data_type, typename traits::non_const_data_type>,
            typename traits::array_layout,
            std::conditional_t<AT == access_target::host, typename traits::host_mirror_space::memory_space, typename traits::memory_space>,
            typename traits::memory_traits>;

    static constexpr unsigned rank = traits::dimension::rank;

    static constexpr unsigned rank_dynamic = traits::dimension::rank_dynamic;

private:
    modified_flag_type m_modified_flag_device;

    modified_flag_type m_modified_flag_host;

    non_const_view_device_type m_view_device;

    non_const_view_host_type m_view_host;

    template <access_target AT>
    [[nodiscard]] auto get_view(AccessTargetTag<AT> /*target*/) const
    {
        if constexpr (AccessTargetTag<AT>::value == access_target::device) {
            return m_view_device;
        } else if constexpr (AccessTargetTag<AT>::value == access_target::host) {
            return m_view_host;
        }
    }

    template <access_target AT>
    auto mark_modified(AccessTargetTag<AT> /*target*/) const noexcept -> void
    {
        assert(is_allocated());

        unsigned const new_flag = std::max(m_modified_flag_host(), m_modified_flag_device()) + 1;

        if constexpr (AccessTargetTag<AT>::value == access_target::device) {
            m_modified_flag_device() = new_flag;
        } else if constexpr (AccessTargetTag<AT>::value == access_target::host) {
            m_modified_flag_host() = new_flag;
        }
    }

    template <access_mode AM, access_target AT>
    [[nodiscard]] auto access(AccessTag<AT, AM> const target_and_mode) const -> view_type<AM, AT>
    {
        assert(is_allocated());

        if constexpr (has_read_access(get_mode(target_and_mode))) {
            synchronize(get_target(target_and_mode));
        }

        if constexpr (has_write_access(get_mode(target_and_mode))) {
            static_assert(!std::is_const_v<typename traits::value_type>, "Cannot get write access on a DualView with a const datatype.");

            mark_modified(get_target(target_and_mode));
        }

        return view_type<AM, AT>(get_view(get_target(target_and_mode)));
    }

public:
    DualView() = default;

    template <typename... Args>
    explicit DualView(std::string const& label, Args... args)
        : m_modified_flag_device("DualView::modified_flag_device")
        , m_modified_flag_host("DualView::modified_flag_host")
        , m_view_device(label, args...)
        , m_view_host(Kokkos::create_mirror_view(m_view_device))
    {
        static_assert(sizeof...(Args) == rank);
        static_assert(std::is_constructible_v<view_device_type, std::string, Args...>);
    }

    template <typename DT, typename... DP>
    explicit DualView(DualView<DT, DP...> const& rhs)
        : m_modified_flag_device(rhs.m_modified_flag_device)
        , m_modified_flag_host(rhs.m_modified_flag_host)
        , m_view_device(rhs.m_view_device)
        , m_view_host(rhs.m_view_host)
    {
        static_assert(std::is_constructible_v<view_device_type, Kokkos::View<DT, DP...> const&>);
    }

    template <typename DT, typename... DP>
    explicit DualView(DualView<DT, DP...>&& rhs) noexcept
        : m_modified_flag_device(std::move(rhs.m_modified_flag_device))
        , m_modified_flag_host(std::move(rhs.m_modified_flag_host))
        , m_view_device(std::move(rhs.m_view_device))
        , m_view_host(std::move(rhs.m_view_host))
    {
        static_assert(std::is_constructible_v<view_device_type, Kokkos::View<DT, DP...>&&>);
    }

    DualView(DualView const& rhs) = default;

    DualView(DualView&& rhs) noexcept = default;

    ~DualView() noexcept = default;

    auto operator=(DualView const& rhs) -> DualView& = default;

    auto operator=(DualView&& rhs) noexcept -> DualView& = default;

    template <typename DT, typename... DP>
    auto operator=(DualView<DT, DP...> const& rhs) -> DualView&
    {
        static_assert(std::is_assignable_v<view_device_type, Kokkos::View<DT, DP...> const&>);
        m_modified_flag_device = rhs.m_modified_flag_device;
        m_modified_flag_host = rhs.m_modified_flag_host;
        m_view_device = rhs.m_view_device;
        m_view_host = rhs.m_view_host;
        return *this;
    }

    template <typename DT, typename... DP>
    auto operator=(DualView<DT, DP...>&& rhs) noexcept -> DualView&
    {
        static_assert(std::is_assignable_v<view_device_type, Kokkos::View<DT, DP...>&&>);
        m_modified_flag_device = std::move(rhs.m_modified_flag_device);
        m_modified_flag_host = std::move(rhs.m_modified_flag_host);
        m_view_device = std::move(rhs.m_view_device);
        m_view_host = std::move(rhs.m_view_host);
        return *this;
    }

    [[nodiscard]] auto label() const -> std::string
    {
        return m_view_device.label();
    }

    [[nodiscard]] auto is_allocated() const noexcept -> bool
    {
        assert(m_modified_flag_device.is_allocated() == m_modified_flag_host.is_allocated());
        assert(m_view_device.is_allocated() == m_view_host.is_allocated());

        return m_view_device.is_allocated();
    }

    template <access_target AT>
    [[nodiscard]] auto need_synchronization(AccessTargetTag<AT> /*target*/) const noexcept -> bool
    {
        assert(is_allocated());

        if constexpr (AccessTargetTag<AT>::value == access_target::device) {
            return m_modified_flag_device() < m_modified_flag_host();
        } else if constexpr (AccessTargetTag<AT>::value == access_target::host) {
            return m_modified_flag_host() < m_modified_flag_device();
        }
    }

    template <access_target AT>
    auto synchronize(AccessTargetTag<AT> const target) const -> void
    {
        assert(is_allocated());

        if (need_synchronization(target)) {
            if (target.value == access_target::device) {
                Kokkos::deep_copy(m_view_device, m_view_host);
            } else {
                Kokkos::deep_copy(m_view_host, m_view_device);
            }
            clear_synchronization_state();
        }
    }

    template <access_mode AM, access_target AT>
    [[nodiscard]] auto operator()(AccessTag<AT, AM> const target_and_mode) const -> view_type<AM, AT>
    {
        return access(target_and_mode);
    }

    template <access_target AT, access_mode AM>
    [[nodiscard]] auto operator()(AccessTargetTag<AT> const /*target*/, AccessModeTag<AM> /*mode*/) const -> view_type<AM, AT>
    {
        return access(AccessTag<AT, AM>());
    }

    template <access_target AT>
    [[nodiscard]] auto operator()(AccessTargetTag<AT> const /*target*/) const -> view_type<access_mode::read_write, AT>
    {
        return access(AccessTag<AT, access_mode::read_write>());
    }

    auto clear_synchronization_state() const noexcept -> void
    {
        assert(is_allocated());

        m_modified_flag_device() = 0U;
        m_modified_flag_host() = 0U;
    }

    [[nodiscard]] auto span() const noexcept -> std::size_t
    {
        return m_view_device.span();
    }

    [[nodiscard]] auto span_is_contiguous() const noexcept -> bool
    {
        return m_view_device.span_is_contiguous();
    }

    [[nodiscard]] auto const_view() const noexcept -> DualView<typename traits::const_data_type, Layout, MemorySpace>
    {
        return DualView<typename traits::const_data_type, Layout, MemorySpace>(*this);
    }

    template <typename IntegralType>
    [[nodiscard]] auto stride(IntegralType const& r) const noexcept -> std::size_t
    {
        static_assert(std::is_integral_v<IntegralType>);
        assert(r >= 0 && static_cast<unsigned>(r) < rank);
        return m_view_device.stride(r);
    }

    template <typename IntegralType>
    [[nodiscard]] auto extent(IntegralType const& r) const noexcept -> std::size_t
    {
        static_assert(std::is_integral_v<IntegralType>);
        assert(r >= 0 && static_cast<unsigned>(r) < rank);
        return m_view_device.extent(r);
    }

    template <typename IntegralType>
    [[nodiscard]] auto extent_int(IntegralType const& r) const noexcept -> int
    {
        static_assert(std::is_integral_v<IntegralType>);
        assert(r >= 0 && static_cast<unsigned>(r) < rank);
        return m_view_device.extent_int(r);
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t
    {
        return m_view_device.size();
    }
};

} // namespace hclpp
