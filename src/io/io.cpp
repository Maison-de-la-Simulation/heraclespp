#include <mpi.h>

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <grid.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <pdi.h>

#include "io.hpp"

namespace novapp
{

void write_pdi_init(int max_iter, int frequency, Grid const& grid, Param const& param)
{
    int mpi_rank = grid.mpi_rank;
    int mpi_size = grid.mpi_size;
    int simu_nfx = param.nfx;

    PDI_multi_expose("init_PDI",
                    "max_iter", &max_iter, PDI_OUT,
                    "frequency", &frequency, PDI_OUT,
                    "nfx", &simu_nfx, PDI_OUT,
                    "mpi_rank", &mpi_rank, PDI_OUT,
                    "mpi_size", &mpi_size, PDI_OUT,
                    //"n_ghost", grid.Nghost.data(), PDI_OUT,
                    //"nx_glob_ng", grid.Nx_glob_ng.data(), PDI_OUT,
                    //"nx_local_ng", grid.Nx_local_ng.data(), PDI_OUT,
                    //"nx_local_wg", grid.Nx_local_wg.data(), PDI_OUT,
                    "start", grid.range.Corner_min.data(), PDI_OUT,
                    "grid_communicator", &grid.comm_cart, PDI_OUT,
                    NULL);
}

void write_pdi(Grid const& grid,
               int iter,
               double t,
               double gamma,
               KDV_double_3d rho,
               KDV_double_4d u,
               KDV_double_3d P,
               KDV_double_3d E,
               KDV_double_1d x,
               KDV_double_1d y,
               KDV_double_1d z,
               KDV_double_4d fx,
               KDV_double_3d T)
{
    int ndim_u = u.extent(3);
    std::cout <<"write dim = " << ndim_u << std::endl;

    sync_host(rho, u, P, E, fx, T, x, y, z);
    PDI_multi_expose("write_file",
                    "iter", &iter, PDI_OUT,
                    "current_time", &t, PDI_OUT,
                    "gamma", &gamma, PDI_OUT,
                    "ndim_u", &ndim_u, PDI_OUT,
                    "n_ghost", grid.Nghost.data(), PDI_OUT,
                    "nx_glob_ng", grid.Nx_glob_ng.data(), PDI_OUT,
                    "nx_local_ng", grid.Nx_local_ng.data(), PDI_OUT,
                    "nx_local_wg", grid.Nx_local_wg.data(), PDI_OUT,
                    "rho", rho.h_view.data(), PDI_OUT,
                    "u", u.h_view.data(), PDI_OUT,
                    "P", P.h_view.data(), PDI_OUT,
                    "E", E.h_view.data(), PDI_OUT,
                    "x", x.h_view.data(), PDI_OUT,
                    "y", y.h_view.data(), PDI_OUT,
                    "z", z.h_view.data(), PDI_OUT,
                    "fx", fx.h_view.data(), PDI_OUT,
                    "T", T.h_view.data(), PDI_OUT,
                    NULL);
}

should_output_fn::should_output_fn(int freq, int iter_max, double time_out)
    : m_freq(freq)
    , m_iter_max(iter_max)
    , m_time_out(time_out)
{
}

bool should_output_fn::operator()(int iter, double current, double dt) const
{
    bool result = (m_freq > 0)
                  && (((iter + 1) >= m_iter_max) || ((iter + 1) % m_freq == 0)
                      || (current + dt >= m_time_out));
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (result && (mpi_rank == 0)) {
        std::cout << std::left << std::setw(80) << std::setfill('*') << "*" << std::endl;
        std::cout << "current iteration " << iter + 1 << " : " << std::endl;
        std::cout << "current time = " << current << " ( ~ " << 100 * (current) / m_time_out << "%)"
                  << std::endl
                  << std::endl;
    }

    return result;
}

void read_pdi(std::string restart_file,
              Grid const& grid,
              Param const& param,
              int& iter,
              double& t,
              KDV_double_3d rho,
              KDV_double_4d u,
              KDV_double_3d P,
              KDV_double_4d fx,
              KDV_double_1d x_glob,
              KDV_double_1d y_glob,
              KDV_double_1d z_glob)
{
    int ndim_u = u.extent(3);
    std::cout <<"read dim = " << ndim_u << std::endl;

    std::array<int, 3> Nghost;
    for (int idim = 0; idim < ndim_u; idim++)
    {
        Nghost[idim] = param.Ng;
    }
    std::array<int, 3> Nx_glob_ng;
    std::array<int, 3> Nx_local_ng;
    std::array<int, 3> Nx_local_wg;

    int filename_size = restart_file.size();
    PDI_multi_expose("read_file",
                    "restart_filename_size", &filename_size, PDI_INOUT,
                    "restart_filename", restart_file.data(), PDI_INOUT,
                    "iter", &iter, PDI_INOUT,
                    "current_time", &t, PDI_INOUT,
                    "ndim_u", &ndim_u, PDI_OUT,
                    "n_ghost", Nghost.data(), PDI_OUT,
                    "nx_glob_ng", Nx_glob_ng.data(), PDI_OUT,
                    "nx_local_ng", Nx_local_ng.data(), PDI_OUT,
                    "nx_local_wg", Nx_local_wg.data(), PDI_OUT,
                    "rho", rho.h_view.data(), PDI_INOUT,
                    "u", u.h_view.data(), PDI_INOUT,
                    "P", P.h_view.data(), PDI_INOUT,
                    "fx", fx.h_view.data(), PDI_INOUT,
                    "x", x_glob.h_view.data(), PDI_INOUT,
                    "y", y_glob.h_view.data(), PDI_INOUT,
                    "z", z_glob.h_view.data(), PDI_INOUT,
                    NULL);
    modify_host(rho, u, P, fx, x_glob, y_glob, z_glob);
}

void writeXML(
        Grid const& grid,
        std::vector<std::pair<int, double>> const& outputs_record,
        KDV_double_1d x,
        KDV_double_1d y,
        KDV_double_1d z)
{
    sync_host(x, y, z);
    if (grid.mpi_rank != 0)
    {
        return;
    }

    std::string const prefix("test");
    std::string const xdmfFilenameFull(prefix + ".xmf");
    std::ofstream xdmfFile(xdmfFilenameFull, std::ofstream::trunc);

    auto const getFilename = [&](int num) {
        std::ostringstream restartNum;
        restartNum << std::setw(8);
        restartNum << std::setfill('0');
        restartNum << num;
        return prefix + '_' + restartNum.str() + ".h5";
    };

    xdmfFile << "<?xml version=\"1.0\"?>\n";
    xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xdmfFile << "<Xdmf Version=\"2.0\">\n";
    xdmfFile << std::string(2, ' ') << "<Domain>\n";
    xdmfFile << std::string(4, ' ');
    xdmfFile << "<Grid";
    xdmfFile << " Name=" << '"' << "TimeSeries" << '"';
    xdmfFile << " GridType=" << '"' << "Collection" << '"';
    xdmfFile << " CollectionType=" << '"' << "Temporal" << '"';
    xdmfFile << ">\n";

    std::array<int, 3> const ncells = grid.Nx_glob_ng;

    int const precision = sizeof(double);

    for (std::pair<int, double> const& it : outputs_record)
    {
        xdmfFile << std::string(6, ' ');
        xdmfFile << "<Grid Name=" << '"' << "output" << '"';
        xdmfFile << " GridType=" << '"' << "Uniform" << '"';
        xdmfFile << ">\n";
        xdmfFile << std::string(8, ' ');
        xdmfFile << "<Time Value=" << '"' << it.second << '"' << "/>\n";

        // topology CoRectMesh
        xdmfFile << std::string(8, ' ');
        xdmfFile << "<Topology";
        xdmfFile << " TopologyType=" << '"' << "3DRectMesh" << '"';
        xdmfFile << " Dimensions=" << '"';
        for (int idim = 2; idim >= 0; --idim)
        {
            xdmfFile << ncells[idim] + 1;
            xdmfFile << (idim == 0 ? "\"" : " ");
        }
        xdmfFile << "/>\n";

        // geometry
        xdmfFile << std::string(8, ' ');
        xdmfFile << "<Geometry";
        xdmfFile << " GeometryType=" << '"' << "VXVYVZ" << '"';
        xdmfFile << ">\n";

        auto nghost = grid.range.Nghost;
        for (int idim = 0; idim < 3; ++idim)
        {
            auto arrays = {x.h_view, y.h_view, z.h_view};
            auto array = arrays.begin()[idim];
            xdmfFile << std::string(10, ' ');
            xdmfFile << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';
            xdmfFile << " Dimensions=" << '"' << array.extent_int(0) - 2 * nghost[idim] << '"';
            xdmfFile << " Format=" << '"' << "XML" << '"';
            xdmfFile << ">\n";
            xdmfFile << std::string(12, ' ');
            for (int ix = nghost[idim]; ix < array.extent_int(0) - 1 - nghost[idim]; ++ix)
            {
                xdmfFile << array(ix);
                xdmfFile << " ";
            }
            xdmfFile << array(array.extent_int(0) - 1 - nghost[idim]) << "\n";
            xdmfFile << std::string(10, ' ') << "</DataItem>\n";
        }

        xdmfFile << std::string(8, ' ') << "</Geometry>\n";

        for (std::string_view var_name : {"rho", "P", "E", "fx"})
        {
            xdmfFile << std::string(8, ' ');
            xdmfFile << "<Attribute";
            xdmfFile << " Center=" << '"' << "Cell" << '"';
            xdmfFile << " Name=" << '"' << var_name << '"';
            xdmfFile << " AttributeType=" << '"' << "Scalar" << '"';
            xdmfFile << ">\n";
            xdmfFile << std::string(10, ' ');
            xdmfFile << "<DataItem";
            xdmfFile << " NumberType=" << '"' << "Float" << '"';
            xdmfFile << " Precision=" << '"' << precision << '"';

            xdmfFile << " Dimensions=\"";
            for (int idim = 2; idim >= 0; --idim)
            {
                xdmfFile << ncells[idim];
                xdmfFile << (idim == 0 ? "\"" : " ");
            }

            xdmfFile << " Format=" << '"' << "HDF" << '"';
            xdmfFile << ">\n";
            xdmfFile << std::string(12, ' ') << getFilename(it.first) << ":/" << var_name
                     << "\n";
            xdmfFile << std::string(10, ' ') << "</DataItem>\n";
            xdmfFile << std::string(8, ' ') << "</Attribute>\n";
        }

        std::vector<std::string> const components {"x", "y", "z"};
        for (std::string_view var_name : {"u"})
        {
            for (int icomp = 0; icomp < ndim; ++icomp)
            {
                xdmfFile << std::string(8, ' ');
                xdmfFile << "<Attribute";
                xdmfFile << " Center=" << '"' << "Cell" << '"';
                xdmfFile << " Name=" << '"' << var_name << components[icomp] << '"';
                xdmfFile << " AttributeType=" << '"' << "Scalar" << '"';
                xdmfFile << ">\n";
                xdmfFile << std::string(10, ' ');
                xdmfFile << "<DataItem";
                xdmfFile << " NumberType=" << '"' << "Float" << '"';
                xdmfFile << " Precision=" << '"' << precision << '"';

                xdmfFile << " Dimensions=\"";
                for (int idim = 2; idim >= 0; --idim)
                {
                    xdmfFile << ncells[idim];
                    xdmfFile << (idim == 0 ? "\"" : " ");
                }

                xdmfFile << " Format=" << '"' << "HDF" << '"';
                xdmfFile << ">\n";
                xdmfFile << std::string(12, ' ') << getFilename(it.first) << ":/" << var_name << components[icomp]
                         << "\n";
                xdmfFile << std::string(10, ' ') << "</DataItem>\n";
                xdmfFile << std::string(8, ' ') << "</Attribute>\n";
            }
        }

        // finalize grid file for the current time step
        xdmfFile << std::string(6, ' ') << "</Grid>\n";
    }

    // finalize Xdmf wrapper file
    xdmfFile << std::string(4, ' ') << "</Grid>\n";
    xdmfFile << std::string(2, ' ') << "</Domain>\n";
    xdmfFile << "</Xdmf>\n";
}

} // namespace novapp
