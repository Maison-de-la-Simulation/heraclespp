#include "io.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pdi.h>
#include <mpi.h>

#include "grid.hpp"
#include "nova_params.hpp"

namespace novapp
{

void write_pdi_init(int max_iter, int frequency, Grid const& grid, Param const& param)
{
    int mpi_rank = grid.mpi_rank;
    int mpi_size = grid.mpi_size;
    int simu_ndim = grid.Ndim;
    int simu_nfx = param.nfx;

    PDI_multi_expose("init_PDI",
                    "max_iter", &max_iter, PDI_OUT,
                    "frequency", &frequency, PDI_OUT,
                    "ndim", &simu_ndim, PDI_OUT,
                    "nfx", &simu_nfx, PDI_OUT,
                    "mpi_rank", &mpi_rank, PDI_OUT,
                    "mpi_size", &mpi_size, PDI_OUT,
                    "n_ghost", grid.Nghost.data(), PDI_OUT,
                    "nx_glob_ng", grid.Nx_glob_ng.data(), PDI_OUT,
                    "nx_local_ng", grid.Nx_local_ng.data(), PDI_OUT,
                    "nx_local_wg", grid.Nx_local_wg.data(), PDI_OUT,
                    "start", grid.range.Corner_min.data(), PDI_OUT,
                    "grid_communicator", &grid.comm_cart, PDI_OUT,
                    NULL);
}

void write_pdi(int iter,
               double t,
               double gamma,
               KDV_double_3d rho,
               KDV_double_4d u,
               KDV_double_3d P, 
               KDV_double_3d E,
               KDV_double_1d x,
               KDV_double_1d y,
               KDV_double_1d z,
               KDV_double_4d fx)
{
    rho.sync_host();
    u.sync_host();
    P.sync_host();
    E.sync_host();
    fx.sync_host();
    x.sync_host();
    y.sync_host();
    z.sync_host();
    PDI_multi_expose("write_file",
                    "iter", &iter, PDI_OUT,
                    "current_time", &t, PDI_OUT,
                    "gamma", &gamma, PDI_OUT,
                    "rho", rho.h_view.data(), PDI_OUT,
                    "u", u.h_view.data(), PDI_OUT,
                    "P", P.h_view.data(), PDI_OUT,
                    "E", E.h_view.data(), PDI_OUT,
                    "x", x.h_view.data(), PDI_OUT,
                    "y", y.h_view.data(), PDI_OUT,
                    "z", z.h_view.data(), PDI_OUT,
                    "fx", fx.h_view.data(), PDI_OUT,
                    NULL);
}

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out)
{
    bool result = (freq > 0) && (((iter+1)>=iter_max) || ((iter+1)%freq==0) || (current+dt>=time_out));
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(result && (mpi_rank==0))
    {
        std::cout<< std::left << std::setw(80) << std::setfill('*') << "*"<<std::endl;
        std::cout<<"current iteration "<<iter+1<<" : "<<std::endl;
        std::cout<<"current time = "<<current<<" ( ~ "<< 100*(current)/time_out <<"%)"<<std::endl<<std::endl ;
    }

    return result;
}

void read_pdi(std::string restart_file,
              KDV_double_3d rho,
              KDV_double_4d u,
              KDV_double_3d P,
              KDV_double_4d fx,
              double &t, int &iter)
{
    int filename_size = restart_file.size();
    PDI_multi_expose("read_file",
                    "restart_filename_size", &filename_size, PDI_INOUT,
                    "restart_filename", restart_file.data(), PDI_INOUT,
                    "iter", &iter, PDI_INOUT,
                    "current_time", &t, PDI_INOUT, 
                    "rho", rho.h_view.data(), PDI_INOUT,
                    "u", u.h_view.data(), PDI_INOUT,
                    "P", P.h_view.data(), PDI_INOUT,
                    "fx", fx.h_view.data(), PDI_INOUT,
                    NULL);
    rho.modify_host();
    u.modify_host();
    P.modify_host();
    fx.modify_host();
}

void writeXML(
        Grid const& grid,
        std::vector<std::pair<int, double>> const& outputs_record,
        KVH_double_1d x,
        KVH_double_1d y,
        KVH_double_1d z)
{
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
            auto arrays = {x, y, z};
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
