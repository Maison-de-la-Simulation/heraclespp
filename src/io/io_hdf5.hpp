#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include <hdf5.h>

namespace novapp {

class raii_h5_hid
{
private:
    hid_t m_id;

    std::function<herr_t(hid_t)> m_close;

public:
    raii_h5_hid(hid_t id, herr_t (*f)(hid_t)) : m_id(id), m_close(f)
    {
        if (m_id < 0 || !m_close) {
            throw std::runtime_error("Nova++ error: creating h5 id failed");
        }
    }

    raii_h5_hid(const raii_h5_hid&) = delete;

    raii_h5_hid(raii_h5_hid&&) = delete;

    ~raii_h5_hid() noexcept
    {
        if (m_id >= 0 && m_close) {
            m_close(m_id);
        }
    }

    raii_h5_hid& operator=(const raii_h5_hid&) = delete;

    raii_h5_hid& operator=(raii_h5_hid&&) = delete;

    hid_t operator*() const noexcept
    {
        return m_id;
    }
};

/// @param[in] file_id A file id created by H5Fopen
/// @param[in] dset_path A c-string representing the path of a dataset inside \p file_id
/// @param[in] expected_extents Expected extents of \p dset_path
template <std::size_t N>
void check_extent_dset(
        raii_h5_hid const& file_id,
        char const* const dset_path,
        std::array<std::size_t, N> const& expected_extents)
{
    raii_h5_hid const dset_id(::H5Dopen(*file_id, dset_path, H5P_DEFAULT), ::H5Dclose);
    raii_h5_hid const dspace(::H5Dget_space(*dset_id), ::H5Sclose);
    int const ndims = ::H5Sget_simple_extent_ndims(*dspace);
    if (ndims != N) {
        throw std::runtime_error("Nova++ error: Expecting a 1d dataset");
    }
    std::array<hsize_t, N> dset_extent {};
    ::H5Sget_simple_extent_dims(*dspace, dset_extent.data(), nullptr);
    if (!std::equal(dset_extent.cbegin(), dset_extent.cend(), expected_extents.cbegin())) {
        throw std::runtime_error("Nova++ error: Extents do not match");
    }
}

} // namespace novapp
