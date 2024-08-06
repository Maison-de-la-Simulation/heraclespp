#include <mpi.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <git_version.hpp>
#include <grid.hpp>
#include <hdf5.h>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <pdi.h>

#include "io.hpp"
#include "io_hdf5.hpp"

namespace {

void write_string_attribute(
        novapp::raii_h5_hid const& file_id,
        char const* const attribute_name,
        std::string_view const attribute_value)
{
    novapp::raii_h5_hid const space_id(::H5Screate(H5S_SCALAR), ::H5Sclose);

    novapp::raii_h5_hid const type_id(::H5Tcopy(H5T_C_S1), ::H5Tclose);
    if (::H5Tset_size(*type_id, attribute_value.size()) < 0) {
        throw std::runtime_error("Nova++ error: defining the size of the datatype failed");
    }
    if (::H5Tset_cset(*type_id, H5T_CSET_UTF8) < 0) {
        throw std::runtime_error("Nova++ error: defining utf-8 character set failed");
    }

    novapp::raii_h5_hid const attr_id(
            ::H5Acreate2(*file_id, attribute_name, *type_id, *space_id, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Aclose);

    if (::H5Awrite(*attr_id, *type_id, attribute_value.data()) < 0) {
        throw std::runtime_error("Nova++ error: writing attribute failed");
    }
}

template <class... Views>
bool span_is_contiguous(Views const&... views)
{
    return (views.span_is_contiguous() && ...);
}

std::string get_output_filename(std::string const& prefix, int const num) {
    std::ostringstream output_filename;
    output_filename << prefix;
    output_filename << '_';
    output_filename << std::setw(8);
    output_filename << std::setfill('0');
    output_filename << num;
    output_filename << ".h5";
    return output_filename.str();
}

std::string_view indent(int const width) noexcept
{
    assert(width > 0);
    assert(width <= 20);
    static constexpr std::string_view spaces("                    ");
    return spaces.substr(0, width);
}

} // namespace

namespace novapp
{

void print_simulation_status(
        std::ostream& os,
        int const iter,
        double const current,
        double const time_out,
        int const output_id)
{
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::stringstream ss;
        ss << std::setfill('*') << std::setw(81) << '\n';
        ss << "current iteration " << iter << " : \n";
        ss << "current time = " << current << " ( ~ " << 100 * current / time_out << "%)\n";
        ss << "file number  = " << output_id << "\n\n";
        os << ss.str();
    }
}

void write_pdi_init(
    Grid const& grid,
    Param const& param)
{
    int const simu_ndim = ndim;
    int const simu_nfx = param.nfx;

    PDI_multi_expose(
        "init_PDI",
        "nullptr", nullptr, PDI_OUT,
        "ndim", &simu_ndim, PDI_OUT,
        "nfx", &simu_nfx, PDI_OUT,
        "n_ghost", grid.Nghost.data(), PDI_OUT,
        "nx_glob_ng", grid.Nx_glob_ng.data(), PDI_OUT,
        "nx_local_ng", grid.Nx_local_ng.data(), PDI_OUT,
        "nx_local_wg", grid.Nx_local_wg.data(), PDI_OUT,
        "start", grid.range.Corner_min.data(), PDI_OUT,
        "grid_communicator", &grid.comm_cart, PDI_OUT,
        "mpi_rank", &grid.mpi_rank, PDI_OUT,
        nullptr);
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
    KDV_double_1d& x,
    KDV_double_1d& y,
    KDV_double_1d& z,
    KDV_double_4d& fx,
    KDV_double_3d& T)
{
    assert(span_is_contiguous(rho, u, P, E, fx, T));
    int const directory_size = directory.size();
    std::string const output_filename = get_output_filename(prefix, output_id);
    int const output_filename_size = output_filename.size();
    sync_host(rho, u, P, E, fx, T, x, y, z);
    PDI_multi_expose(
        "write_replicated_data",
        "nullptr", nullptr, PDI_OUT,
        "directory_size", &directory_size, PDI_OUT,
        "directory", directory.data(), PDI_OUT,
        "output_filename_size", &output_filename_size, PDI_OUT,
        "output_filename", output_filename.data(), PDI_OUT,
        "output_id", &output_id, PDI_OUT,
        "iter_output_id", &iter_output_id, PDI_OUT,
        "time_output_id", &time_output_id, PDI_OUT,
        "iter", &iter, PDI_OUT,
        "current_time", &t, PDI_OUT,
        "gamma", &gamma, PDI_OUT,
        "x", x.h_view.data(), PDI_OUT,
        "y", y.h_view.data(), PDI_OUT,
        "z", z.h_view.data(), PDI_OUT,
        nullptr);
    PDI_multi_expose(
        "write_distributed_data",
        "nullptr", nullptr, PDI_OUT,
        "directory_size", &directory_size, PDI_OUT,
        "directory", directory.data(), PDI_OUT,
        "output_filename_size", &output_filename_size, PDI_OUT,
        "output_filename", output_filename.data(), PDI_OUT,
        "rho", rho.h_view.data(), PDI_OUT,
        "u", u.h_view.data(), PDI_OUT,
        "P", P.h_view.data(), PDI_OUT,
        "E", E.h_view.data(), PDI_OUT,
        "fx", fx.h_view.data(), PDI_OUT,
        "T", T.h_view.data(), PDI_OUT,
        nullptr);
    if (grid.mpi_rank == 0)
    {
        raii_h5_hid const file_id(::H5Fopen((directory + '/' + output_filename).c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose);
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
    KDV_double_1d& x_glob,
    KDV_double_1d& y_glob,
    KDV_double_1d& z_glob)
{
    assert(span_is_contiguous(rho, u, P, fx));

    raii_h5_hid const file_id(::H5Fopen(restart_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), ::H5Fclose);
    check_extent_dset(file_id, "/rho", std::array {rho.extent(2), rho.extent(1), rho.extent(0)});
    check_extent_dset(file_id, "/u_x", std::array {u.extent(2), u.extent(1), u.extent(0)});
    if (ndim > 1)
    {
        check_extent_dset(file_id, "/u_y", std::array {u.extent(2), u.extent(1), u.extent(0)});
    }
    if (ndim > 2)
    {
        check_extent_dset(file_id, "/u_z", std::array {u.extent(2), u.extent(1), u.extent(0)});
    }
    check_extent_dset(file_id, "/P", std::array {P.extent(2), P.extent(1), P.extent(0)});
    if (fx.extent(3) > 0)
    {
        check_extent_dset(file_id, "/fx", std::array {fx.extent(3), fx.extent(2), fx.extent(1), fx.extent(0)});
    }
    check_extent_dset(file_id, "/x", std::array {x_glob.extent(0)});
    check_extent_dset(file_id, "/y", std::array {y_glob.extent(0)});
    check_extent_dset(file_id, "/z", std::array {z_glob.extent(0)});

    int const filename_size = restart_file.size();
    PDI_multi_expose(
        "read_file",
        "nullptr", nullptr, PDI_OUT,
        "restart_filename_size", &filename_size, PDI_OUT,
        "restart_filename", restart_file.data(), PDI_OUT,
        "output_id", &output_id, PDI_INOUT,
        "iter_output_id", &iter_output_id, PDI_INOUT,
        "time_output_id", &time_output_id, PDI_INOUT,
        "iter", &iter, PDI_INOUT,
        "current_time", &t, PDI_INOUT,
        "rho", rho.h_view.data(), PDI_INOUT,
        "u", u.h_view.data(), PDI_INOUT,
        "P", P.h_view.data(), PDI_INOUT,
        "fx", fx.h_view.data(), PDI_INOUT,
        "x", x_glob.h_view.data(), PDI_INOUT,
        "y", y_glob.h_view.data(), PDI_INOUT,
        "z", z_glob.h_view.data(), PDI_INOUT,
        nullptr);
    modify_host(rho, u, P, fx, x_glob, y_glob, z_glob);
}

XmlWriter::XmlWriter(std::string directory, std::string prefix, int const nfx)
    : m_directory(std::move(directory))
    , m_prefix(std::move(prefix))
    , m_var_names({"rho", "P", "E", "T"})
{
    if (nfx > 0) {
        m_var_names.emplace_back("fx");
    }
    std::array<std::string_view, 3> const velocity {"ux", "uy", "uz"};
    for (int idim = 0; idim < ndim; ++idim) {
        m_var_names.emplace_back(velocity[idim]);
    }
}

void XmlWriter::operator()(
        Grid const& grid,
        int const output_id,
        std::vector<std::pair<int, double>> const& outputs_record,
        KDV_double_1d& x,
        KDV_double_1d& y,
        KDV_double_1d& z) const
{
    sync_host(x, y, z);
    if (grid.mpi_rank != 0)
    {
        return;
    }

    std::ofstream xdmfFile(m_directory + "/" + m_prefix + ".xmf", std::ofstream::trunc);

    xdmfFile << "<?xml version=\"1.0\"?>\n";
    xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xdmfFile << "<Xdmf Version=\"2.0\">\n";
    xdmfFile << indent(2) << "<Domain>\n";
    xdmfFile << indent(4) << "<Grid";
    xdmfFile << " Name=" << '"' << "TimeSeries" << '"';
    xdmfFile << " GridType=" << '"' << "Collection" << '"';
    xdmfFile << " CollectionType=" << '"' << "Temporal" << '"';
    xdmfFile << ">\n";

    std::array<int, 3> const ncells = grid.Nx_glob_ng;

    int const precision = sizeof(double);

    int const first_output_id = output_id + 1 - outputs_record.size();
    for (std::size_t i = 0; i < outputs_record.size(); ++i)
    {
        xdmfFile << indent(6) << "<Grid";
        xdmfFile << " Name=" << '"' << "output" << '"';
        xdmfFile << " GridType=" << '"' << "Uniform" << '"';
        xdmfFile << ">\n";
        xdmfFile << indent(8) << "<Time";
        xdmfFile << " Value=" << '"' << outputs_record[i].second << '"';
        xdmfFile << "/>\n";

        // topology CoRectMesh
        xdmfFile << indent(8) << "<Topology";
        xdmfFile << " TopologyType=" << '"' << "3DRectMesh" << '"';
        xdmfFile << " Dimensions=" << '"';
        for (int idim = 2; idim >= 0; --idim)
        {
            xdmfFile << ncells[idim] + 1;
            xdmfFile << (idim == 0 ? '"' : ' ');
        }
        xdmfFile << "/>\n";

        // geometry
        xdmfFile << indent(8) << "<Geometry";
        xdmfFile << " GeometryType=" << '"' << "VXVYVZ" << '"';
        xdmfFile << ">\n";

        std::string const output_filename = get_output_filename(m_prefix, first_output_id + i);
        std::array const axes_arrays {x.h_view, y.h_view, z.h_view};
        std::array const axes_labels {"x_ng", "y_ng", "z_ng"};
        for (int idim = 0; idim < 3; ++idim)
        {
            xdmfFile << indent(10) << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';
            xdmfFile << " Dimensions=" << '"' << axes_arrays[idim].extent_int(0) - 2 * grid.Nghost[idim] << '"';
            xdmfFile << " Format=" << '"' << "HDF" << '"';
            xdmfFile << ">\n";
            xdmfFile << indent(12) << output_filename << ":/" << axes_labels[idim]
                     << '\n';
            xdmfFile << indent(10) << "</DataItem>\n";
        }

        xdmfFile << indent(8) << "</Geometry>\n";

        for (std::string const& var_name : m_var_names)
        {
            xdmfFile << indent(8) << "<Attribute";
            xdmfFile << " Center=" << '"' << "Cell" << '"';
            xdmfFile << " Name=" << '"' << var_name << '"';
            xdmfFile << " AttributeType=" << '"' << "Scalar" << '"';
            xdmfFile << ">\n";
            xdmfFile << indent(10) << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';

            xdmfFile << " Dimensions=" << '"';
            for (int idim = 2; idim >= 0; --idim)
            {
                xdmfFile << ncells[idim];
                xdmfFile << (idim == 0 ? '"' : ' ');
            }

            xdmfFile << " Format=" << '"' << "HDF" << '"';
            xdmfFile << ">\n";
            xdmfFile << indent(12) << output_filename << ":/" << var_name
                     << '\n';
            xdmfFile << indent(10) << "</DataItem>\n";
            xdmfFile << indent(8) << "</Attribute>\n";
        }

        // finalize grid file for the current time step
        xdmfFile << indent(6) << "</Grid>\n";
    }

    // finalize Xdmf wrapper file
    xdmfFile << indent(4) << "</Grid>\n";
    xdmfFile << indent(2) << "</Domain>\n";
    xdmfFile << "</Xdmf>\n";
}

} // namespace novapp
