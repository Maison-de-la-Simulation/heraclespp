// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <git_version.hpp>
#include <grid.hpp>
#include <hdf5.h>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <param.hpp>
#include <pdi.h>
#include <range.hpp>

#include "io.hpp"
#include "io_hdf5.hpp"

namespace {

void write_string_attribute(hclpp::RaiiH5Hid const& file_id, char const* const attribute_name, std::string_view const attribute_value)
{
    hclpp::RaiiH5Hid const space_id(::H5Screate(H5S_SCALAR), ::H5Sclose);

    hclpp::RaiiH5Hid const type_id(::H5Tcopy(H5T_C_S1), ::H5Tclose);
    if (::H5Tset_size(*type_id, attribute_value.size()) < 0) {
        throw std::runtime_error("HERACLES++ error: defining the size of the datatype failed");
    }
    if (::H5Tset_cset(*type_id, H5T_CSET_UTF8) < 0) {
        throw std::runtime_error("HERACLES++ error: defining utf-8 character set failed");
    }

    hclpp::RaiiH5Hid const attr_id(::H5Acreate2(*file_id, attribute_name, *type_id, *space_id, H5P_DEFAULT, H5P_DEFAULT), ::H5Aclose);

    if (::H5Awrite(*attr_id, *type_id, attribute_value.data()) < 0) {
        throw std::runtime_error("HERACLES++ error: writing attribute failed");
    }
}

template <class... Views>
bool span_is_contiguous(Views const&... views)
{
    return (views.span_is_contiguous() && ...);
}

std::string get_output_filename(std::string const& prefix, std::size_t const num)
{
    static constexpr int fill_width = 8;

    std::ostringstream output_filename;
    output_filename << prefix;
    output_filename << '_';
    output_filename << std::setw(fill_width);
    output_filename << std::setfill('0');
    output_filename << num;
    output_filename << ".h5";
    return output_filename.str();
}

class IndentFn
{
private:
    std::size_t m_indent_level = 0;

    std::size_t m_max_level;

    std::size_t m_indent_width;

    std::vector<char> m_max_spaces;

    [[nodiscard]] std::string_view to_string_view() const
    {
        return std::string_view(m_max_spaces.data(), m_max_spaces.size()).substr(0, m_indent_width * m_indent_level);
    }

public:
    explicit IndentFn(std::size_t const indent_width = 2, std::size_t const max_level = 10)
        : m_max_level(max_level)
        , m_indent_width(indent_width)
        , m_max_spaces(indent_width * max_level, ' ')
    {
    }

    IndentFn& push()
    {
        if (m_indent_level == m_max_level) {
            throw std::runtime_error("Level out of bound");
        }
        ++m_indent_level;
        return *this;
    }

    IndentFn& pop()
    {
        if (m_indent_level == 0) {
            throw std::runtime_error("Level out of bound");
        }
        --m_indent_level;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, IndentFn const& indent)
    {
        return os << indent.to_string_view();
    }
};

} // namespace

namespace hclpp {

void print_simulation_status(std::ostream& os, int const iter, double const current, double const time_out, int const output_id)
{
    static constexpr int fill_width = 81;
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::stringstream ss;
        ss << std::setfill('*') << std::setw(fill_width) << '\n';
        ss << "current iteration " << iter << " : \n";
        ss << "current time = " << current << " ( ~ " << 100. * current / time_out << "%)\n";
        ss << "file number  = " << output_id << "\n\n";
        os << ss.str();
    }
}

void write_pdi_init(Grid const& grid, Param const& param)
{
    int const simu_ndim = ndim;
    int const simu_nfx = param.nfx;

    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    PDI_multi_expose(
            "init_PDI",
            "nullptr",
            nullptr,
            PDI_OUT,
            "ndim",
            &simu_ndim,
            PDI_OUT,
            "nfx",
            &simu_nfx,
            PDI_OUT,
            "n_ghost",
            grid.Nghost.data(),
            PDI_OUT,
            "nx_glob_ng",
            grid.Nx_glob_ng.data(),
            PDI_OUT,
            "nx_local_ng",
            grid.Nx_local_ng.data(),
            PDI_OUT,
            "nx_local_wg",
            grid.Nx_local_wg.data(),
            PDI_OUT,
            "start",
            grid.range.Corner_min.data(),
            PDI_OUT,
            "grid_communicator",
            &grid.comm_cart,
            PDI_OUT,
            "mpi_rank",
            &grid.mpi_rank,
            PDI_OUT,
            nullptr);
    // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
}

void write_pdi(
        std::string const& directory,
        std::string const& prefix,
        int const output_id,
        int const iter_output_id,
        int const time_output_id,
        int const iter,
        double const t,
        double const gamma,
        Grid const& grid,
        KDV_double_3d& rho,
        KDV_double_4d& u,
        KDV_double_3d& P,
        KDV_double_3d& E,
        KDV_double_1d& x0,
        KDV_double_1d& x1,
        KDV_double_1d& x2,
        KDV_double_4d& fx,
        KDV_double_3d& T)
{
    assert(span_is_contiguous(rho, u, P, E, fx, T));
    int const directory_size = int_cast<int>(directory.size());
    std::string const output_filename = get_output_filename(prefix, output_id);
    int const output_filename_size = int_cast<int>(output_filename.size());
    sync_host(rho, u, P, E, fx, T, x0, x1, x2);
    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    PDI_multi_expose(
            "write_replicated_data",
            "nullptr",
            nullptr,
            PDI_OUT,
            "directory_size",
            &directory_size,
            PDI_OUT,
            "directory",
            directory.data(),
            PDI_OUT,
            "output_filename_size",
            &output_filename_size,
            PDI_OUT,
            "output_filename",
            output_filename.data(),
            PDI_OUT,
            "output_id",
            &output_id,
            PDI_OUT,
            "iter_output_id",
            &iter_output_id,
            PDI_OUT,
            "time_output_id",
            &time_output_id,
            PDI_OUT,
            "iter",
            &iter,
            PDI_OUT,
            "current_time",
            &t,
            PDI_OUT,
            "gamma",
            &gamma,
            PDI_OUT,
            "x0",
            x0.view_host().data(),
            PDI_OUT,
            "x1",
            x1.view_host().data(),
            PDI_OUT,
            "x2",
            x2.view_host().data(),
            PDI_OUT,
            nullptr);
    // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    PDI_multi_expose(
            "write_distributed_data",
            "nullptr",
            nullptr,
            PDI_OUT,
            "directory_size",
            &directory_size,
            PDI_OUT,
            "directory",
            directory.data(),
            PDI_OUT,
            "output_filename_size",
            &output_filename_size,
            PDI_OUT,
            "output_filename",
            output_filename.data(),
            PDI_OUT,
            "rho",
            rho.view_host().data(),
            PDI_OUT,
            "u",
            u.view_host().data(),
            PDI_OUT,
            "P",
            P.view_host().data(),
            PDI_OUT,
            "E",
            E.view_host().data(),
            PDI_OUT,
            "T",
            T.view_host().data(),
            PDI_OUT,
            nullptr);
    for (int ifx = 0; ifx < fx.extent_int(3); ++ifx) {
        PDI_multi_expose(
                "write_fx",
                "nullptr",
                nullptr,
                PDI_OUT,
                "directory_size",
                &directory_size,
                PDI_OUT,
                "directory",
                directory.data(),
                PDI_OUT,
                "output_filename_size",
                &output_filename_size,
                PDI_OUT,
                "output_filename",
                output_filename.data(),
                PDI_OUT,
                "ifx",
                &ifx,
                PDI_OUT,
                "fx",
                fx.view_host().data(),
                PDI_OUT,
                nullptr);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    if (grid.mpi_rank == 0) {
        RaiiH5Hid const file_id(::H5Fopen((directory + '/' + output_filename).c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose);
        write_string_attribute(file_id, "git_build_string", git_build_string);
        write_string_attribute(file_id, "git_branch", git_branch);
        write_string_attribute(file_id, "compile_date", compile_date);
        write_string_attribute(file_id, "compile_time", compile_time);
    }
}

void read_pdi(
        std::string const& restart_file,
        int& output_id,
        int& iter_output_id,
        int& time_output_id,
        int& iter,
        double& t,
        KDV_double_3d& rho,
        KDV_double_4d& u,
        KDV_double_3d& P,
        KDV_double_4d& fx,
        KDV_double_1d& x0_glob,
        KDV_double_1d& x1_glob,
        KDV_double_1d& x2_glob)
{
    assert(span_is_contiguous(rho, u, P, fx));

    RaiiH5Hid const file_id(::H5Fopen(restart_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), ::H5Fclose);
    // check_extent_dset(file_id, "/rho", std::array {rho.extent(2), rho.extent(1), rho.extent(0)});
    // check_extent_dset(file_id, "/u_x0", std::array {u.extent(2), u.extent(1), u.extent(0)});
    // if (ndim > 1)
    // {
    //     check_extent_dset(file_id, "/u_x1", std::array {u.extent(2), u.extent(1), u.extent(0)});
    // }
    // if (ndim > 2)
    // {
    //     check_extent_dset(file_id, "/u_x2", std::array {u.extent(2), u.extent(1), u.extent(0)});
    // }
    // check_extent_dset(file_id, "/P", std::array {P.extent(2), P.extent(1), P.extent(0)});
    // if (fx.extent(3) > 0)
    // {
    //     check_extent_dset(file_id, "/fx", std::array {fx.extent(3), fx.extent(2), fx.extent(1), fx.extent(0)});
    // }
    check_extent_dset(file_id, "/x0", std::array {x0_glob.extent(0)});
    check_extent_dset(file_id, "/x1", std::array {x1_glob.extent(0)});
    check_extent_dset(file_id, "/x2", std::array {x2_glob.extent(0)});

    int const filename_size = int_cast<int>(restart_file.size());
    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    PDI_multi_expose(
            "read_file",
            "nullptr",
            nullptr,
            PDI_OUT,
            "restart_filename_size",
            &filename_size,
            PDI_OUT,
            "restart_filename",
            restart_file.data(),
            PDI_OUT,
            "output_id",
            &output_id,
            PDI_INOUT,
            "iter_output_id",
            &iter_output_id,
            PDI_INOUT,
            "time_output_id",
            &time_output_id,
            PDI_INOUT,
            "iter",
            &iter,
            PDI_INOUT,
            "current_time",
            &t,
            PDI_INOUT,
            "rho",
            rho.view_host().data(),
            PDI_INOUT,
            "u",
            u.view_host().data(),
            PDI_INOUT,
            "P",
            P.view_host().data(),
            PDI_INOUT,
            "x0",
            x0_glob.view_host().data(),
            PDI_INOUT,
            "x1",
            x1_glob.view_host().data(),
            PDI_INOUT,
            "x2",
            x2_glob.view_host().data(),
            PDI_INOUT,
            nullptr);
    for (int ifx = 0; ifx < fx.extent_int(3); ++ifx) {
        PDI_multi_expose(
                "read_fx",
                "nullptr",
                nullptr,
                PDI_OUT,
                "restart_filename_size",
                &filename_size,
                PDI_OUT,
                "restart_filename",
                restart_file.data(),
                PDI_OUT,
                "ifx",
                &ifx,
                PDI_OUT,
                "fx",
                fx.view_host().data(),
                PDI_INOUT,
                nullptr);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    modify_host(rho, u, P, fx, x0_glob, x1_glob, x2_glob);
}

XmlWriter::XmlWriter(std::string directory, std::string prefix, int const nfx)
    : m_directory(std::move(directory))
    , m_prefix(std::move(prefix))
    , m_var_names({"rho", "P", "E", "T"})
{
    for (int ifx = 0; ifx < nfx; ++ifx) {
        m_var_names.emplace_back("fx" + std::to_string(ifx));
    }
    std::array<std::string_view, 3> const velocity {"ux0", "ux1", "ux2"};
    for (int idim = 0; idim < ndim; ++idim) {
        m_var_names.emplace_back(velocity[idim]);
    }
}

void XmlWriter::operator()(
        Grid const& grid,
        int const output_id,
        std::vector<std::pair<int, double>> const& outputs_record,
        KDV_double_1d& x0,
        KDV_double_1d& x1,
        KDV_double_1d& x2) const
{
    sync_host(x0, x1, x2);
    if (grid.mpi_rank != 0) {
        return;
    }

    IndentFn indent;

    std::ofstream xdmfFile(m_directory + '/' + m_prefix + ".xmf", std::ofstream::trunc);

    xdmfFile << indent << "<?xml version=\"1.0\"?>\n";
    xdmfFile << indent << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xdmfFile << indent << "<Xdmf Version=\"2.0\">\n";
    indent.push();
    xdmfFile << indent << "<Domain>\n";
    indent.push();
    xdmfFile << indent << "<Grid";
    xdmfFile << " Name=" << '"' << "TimeSeries" << '"';
    xdmfFile << " GridType=" << '"' << "Collection" << '"';
    xdmfFile << " CollectionType=" << '"' << "Temporal" << '"';
    xdmfFile << ">\n";

    std::array<int, 3> const ncells = grid.Nx_glob_ng;

    int const precision = sizeof(double);

    std::size_t const first_output_id = output_id + 1 - outputs_record.size();
    for (std::size_t i = 0; i < outputs_record.size(); ++i) {
        indent.push();
        xdmfFile << indent << "<Grid";
        xdmfFile << " Name=" << '"' << "output" << '"';
        xdmfFile << " GridType=" << '"' << "Uniform" << '"';
        xdmfFile << ">\n";

        indent.push();
        xdmfFile << indent << "<Time";
        xdmfFile << " Value=" << '"' << outputs_record[i].second << '"';
        xdmfFile << "/>\n";

        // topology CoRectMesh
        xdmfFile << indent << "<Topology";
        xdmfFile << " TopologyType=" << '"' << "3DRectMesh" << '"';
        xdmfFile << " Dimensions=" << '"';
        for (int idim = 2; idim >= 0; --idim) {
            xdmfFile << ncells[idim] + 1;
            xdmfFile << (idim == 0 ? '"' : ' ');
        }
        xdmfFile << "/>\n";

        // geometry
        xdmfFile << indent << "<Geometry";
        xdmfFile << " GeometryType=" << '"' << "VXVYVZ" << '"';
        xdmfFile << ">\n";

        std::string const output_filename = get_output_filename(m_prefix, first_output_id + i);
        std::array const axes_arrays {x0.view_host(), x1.view_host(), x2.view_host()};
        std::array const axes_labels {"x0_ng", "x1_ng", "x2_ng"};
        for (int idim = 0; idim < 3; ++idim) {
            indent.push();
            xdmfFile << indent << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';
            xdmfFile << " Dimensions=" << '"' << axes_arrays[idim].extent_int(0) - (2 * grid.Nghost[idim]) << '"';
            xdmfFile << " Format=" << '"' << "HDF" << '"';
            xdmfFile << ">\n";
            indent.push();
            xdmfFile << indent << output_filename << ":/" << axes_labels[idim] << '\n';
            indent.pop();
            xdmfFile << indent << "</DataItem>\n";
            indent.pop();
        }

        xdmfFile << indent << "</Geometry>\n";

        for (std::string const& var_name : m_var_names) {
            xdmfFile << indent << "<Attribute";
            xdmfFile << " Center=" << '"' << "Cell" << '"';
            xdmfFile << " Name=" << '"' << var_name << '"';
            xdmfFile << " AttributeType=" << '"' << "Scalar" << '"';
            xdmfFile << ">\n";
            indent.push();
            xdmfFile << indent << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';

            xdmfFile << " Dimensions=" << '"';
            for (int idim = 2; idim >= 0; --idim) {
                xdmfFile << ncells[idim];
                xdmfFile << (idim == 0 ? '"' : ' ');
            }

            xdmfFile << " Format=" << '"' << "HDF" << '"';
            xdmfFile << ">\n";
            indent.push();
            xdmfFile << indent << output_filename << ":/" << var_name << '\n';
            indent.pop();
            xdmfFile << indent << "</DataItem>\n";
            indent.pop();
            xdmfFile << indent << "</Attribute>\n";
        }

        indent.pop();
        // finalize grid file for the current time step
        xdmfFile << indent << "</Grid>\n";
        indent.pop();
    }

    // finalize Xdmf wrapper file
    xdmfFile << indent << "</Grid>\n";
    indent.pop();
    xdmfFile << indent << "</Domain>\n";
    indent.pop();
    xdmfFile << indent << "</Xdmf>\n";
}

} // namespace hclpp
