// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include <hdf5.h>

namespace hclpp {

class RaiiH5Hid
{
private:
    hid_t m_id;

    std::function<herr_t(hid_t)> m_close;

public:
    RaiiH5Hid(hid_t id, herr_t (*f)(hid_t)) : m_id(id), m_close(f)
    {
        if (m_id < 0 || !m_close) {
            throw std::runtime_error("HERACLES++ error: creating h5 id failed");
        }
    }

    RaiiH5Hid(RaiiH5Hid const&) = delete;

    RaiiH5Hid(RaiiH5Hid&&) = delete;

    ~RaiiH5Hid() noexcept
    {
        if (m_id >= 0 && m_close) {
            m_close(m_id);
        }
    }

    auto operator=(RaiiH5Hid const&) -> RaiiH5Hid& = delete;

    auto operator=(RaiiH5Hid&&) -> RaiiH5Hid& = delete;

    auto operator*() const noexcept -> hid_t
    {
        return m_id;
    }
};

/// @param[in] file_id A file id created by H5Fopen
/// @param[in] dset_path A c-string representing the path of a dataset inside \p file_id
/// @param[in] expected_extents Expected extents of \p dset_path
template <std::size_t N>
void check_extent_dset(RaiiH5Hid const& file_id, char const* const dset_path, std::array<std::size_t, N> const& expected_extents)
{
    RaiiH5Hid const dset_id(::H5Dopen(*file_id, dset_path, H5P_DEFAULT), ::H5Dclose);
    RaiiH5Hid const dspace(::H5Dget_space(*dset_id), ::H5Sclose);
    int const ndims = ::H5Sget_simple_extent_ndims(*dspace);
    if (ndims != N) {
        throw std::runtime_error("HERACLES++ error: Expecting a 1d dataset");
    }
    std::array<hsize_t, N> dset_extent {};
    ::H5Sget_simple_extent_dims(*dspace, dset_extent.data(), nullptr);
    if (!std::ranges::equal(dset_extent, expected_extents)) {
        throw std::runtime_error("HERACLES++ error: Extents do not match");
    }
}

} // namespace hclpp
