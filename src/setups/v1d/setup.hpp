#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>
#include <Kokkos_Random.hpp>

#include <inih/INIReader.hpp>

#include "broadcast.hpp"
#include "default_boundary_setup.hpp"
#include "default_user_step.hpp"
#include "eos.hpp"
#include <grid.hpp>
#include <grid_type.hpp>
#include "initialization_interface.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "nova_params.hpp"
#include <range.hpp>
#include <pdi.h>
#include <string>

namespace novapp
{

class ParamSetup
{
public:
    std::string init_filename;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
    double xchoc;

    explicit ParamSetup(INIReader const& reader)
        : init_filename(reader.Get("problem", "init_file", ""))
        , ymin(reader.GetReal("Grid", "ymin", 0.0))
        , ymax(reader.GetReal("Grid", "ymax", 1.0))
        , zmin(reader.GetReal("Grid", "zmin", 0.0))
        , zmax(reader.GetReal("Grid", "zmax", 1.0))
        , xchoc(reader.GetReal("Initialisation", "xchoc", 1.0))
        {}
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    EOS m_eos;
    Grid m_grid;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        EOS const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        KV_double_4d const& fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        KDV_double_1d rho_1d("rho_1d", m_grid.Nx_local_ng[0]);
        KDV_double_1d u_1d("u_1d", m_grid.Nx_local_ng[0]);
        KDV_double_1d P_1d("P_1d", m_grid.Nx_local_ng[0]);
        KDV_double_2d fx_1d("fx_1d", m_grid.Nx_local_ng[0], fx.extent_int(3));

        int filename_size = m_param_setup.init_filename.size();
        PDI_multi_expose(
            "read_hydro_1d",
            "init_filename_size", &filename_size, PDI_INOUT,
            "init_filename", m_param_setup.init_filename.data(), PDI_INOUT,
            "rho_1d", rho_1d.h_view.data(), PDI_INOUT,
            "u_1d", u_1d.h_view.data(), PDI_INOUT,
            "P_1d", P_1d.h_view.data(), PDI_INOUT,
            "fx_1d", fx_1d.h_view.data(), PDI_INOUT,
            NULL);
        modify_host(rho_1d, u_1d, P_1d, fx_1d);
        sync_device(rho_1d, u_1d, P_1d, fx_1d);

        broadcast(range, rho_1d.d_view, rho);
        broadcast(range, u_1d.d_view, Kokkos::subview(u, ALL, ALL, ALL, 0));
        for (int idim = 1; idim < u.extent_int(3); ++idim)
        {
            broadcast(range, 0, Kokkos::subview(u, ALL, ALL, ALL, idim));
        }
        broadcast(range, P_1d.d_view, P);
        for(int ifx = 0; ifx < fx.extent_int(3); ++ifx)
        {
            broadcast(range, Kokkos::subview(fx_1d.d_view, ALL, ifx), Kokkos::subview(fx, ALL, ALL, ALL, ifx));
        }

        // perturbation
        auto const& param_setup = m_param_setup;
        auto const& grid = m_grid;
        auto const xc = grid.x_center;
        auto const yc = grid.y_center;
        auto const zc = grid.z_center;
        auto const x = grid.x;

        int ny = rho.extent_int(1) - 4;
        int nz = rho.extent_int(2) - 4;
        //double xchoc = 6.1E9; // tini = 500
        //double xchoc = 5.5E8; // tini = 50

        Kokkos::Random_XorShift64_Pool<> random_pool(54321);
        int kx1 = 60;
        int kx2 = 80;

        // perturbation

        Kokkos::parallel_for(
            "V1D_perturb_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double petrurb_r = 0;
                double perturb_th_ph = 0;
                double noise = 0;

                if (xc(i) < param_setup.xchoc)
                {
                    // background noise
                    auto generator = random_pool.get_state();
                    noise = generator.drand(-1.0, 1.0);

                    // modes for theta and phi
                    double A = 0.015;
                    for (int n = 0; n < 10; ++n)
                    {
                        double kth = ny / (generator.drand(0.0, 1.0) * 50);
                        double kph = nz / (generator.drand(0.0, 1.0) * 50);
                        double alpha1 = (ny + nz) * 10 * generator.drand(0.0, 1.0);
                        double alpha2 = (ny + nz) * 10 * generator.drand(0.0, 1.0);

                        if (ndim == 1)
                        {
                            perturb_th_ph = 0;
                        }
                        if (ndim == 2)
                        {
                            perturb_th_ph += A * Kokkos::sin(j / kth + alpha1);
                        }
                        if (ndim == 3)
                        {
                            perturb_th_ph += A * Kokkos::sin(j / kth + alpha1) * Kokkos::cos(k / kph + alpha2);
                        }
                    }

                    // r perturbation
		    double xmin = x(grid.Nghost[0]);
                    double x0 = param_setup.xchoc - xmin;
                    double sigma = 0.1 * x0 * x0;
                    petrurb_r = Kokkos::exp(-(xc(i) - x0) * (xc(i) - x0) / sigma) * Kokkos::cos(kx1 * xc(i) + kx2 * xc(i));

                    random_pool.free_state(generator);
                }

                double perturb = 0.1 * petrurb_r + perturb_th_ph;
                rho(i, j, k) = rho(i, j, k) * (1 + perturb + 0.01 * noise);
                u(i, j, k, 0) = u(i, j, k, 0) * (1 + perturb + 0.01 * noise);
            });

        // other type of perturbation

        /*  double kx = 60;
        int ny = 20;
        int nz = 20;
        double ky = (2 * units::pi * ny) / (param_setup.ymax - param_setup.ymin);
        double kz = (2 * units::pi * nz) / (param_setup.zmax - param_setup.zmin);
        Kokkos::Random_XorShift64_Pool<> random_pool(12345 + grid.mpi_rank);

        Kokkos::parallel_for(
            "V1D_perturb_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double perturb = 0;
                double noise = 0;

                if (xc(i) < param_setup.xchoc)
                {
                    auto generator = random_pool.get_state();
                    noise = generator.drand(-1.0, 1.0);
                    random_pool.free_state(generator);

                    double x0 = param_setup.xchoc - xmin;
                    double sigma = 0.1 * x0 * x0;
                    perturb = Kokkos::exp(-(xc(i) - x0) * (xc(i) - x0) / sigma) * Kokkos::cos(kx * xc(i));
                    if (ndim == 2)
                    {
                        perturb *= Kokkos::sin(ky * yc(j));
                    }
                    if (ndim == 3)
                    {
                        perturb *= Kokkos::sin(ky * yc(j)) * Kokkos::cos(kz * zc(k));
                    }
                }

                rho(i, j, k) *= 1 + 0.1 * perturb + 0.01 * noise;
                u(i, j, k, 0) *= 1 + 0.1 * perturb + 0.01 * noise;
            }); */
    }
};

class GridSetup : public IGridType
{
private :
    Param m_param;

public:
    explicit GridSetup(Param const& param)
        : m_param(param)
    {
    }

    void execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d const& x_glob,
        KVH_double_1d const& y_glob,
        KVH_double_1d const& z_glob) const final
    {
        std::string init_file = m_param.reader.Get("problem", "init_file", "");
        int filename_size = init_file.size();
        PDI_multi_expose(
            "read_mesh_1d",
            "init_filename_size", &filename_size, PDI_INOUT,
            "init_filename", init_file.data(), PDI_INOUT,
            "x", x_glob.data(), PDI_INOUT,
            NULL);

        compute_regular_mesh_1d(y_glob, Nghost[1], m_param.ymin, (m_param.ymax - m_param.ymin) / Nx_glob_ng[1]);
        compute_regular_mesh_1d(z_glob, Nghost[2], m_param.zmin, (m_param.zmax - m_param.zmin) / Nx_glob_ng[2]);
    }
};

} // namespace novapp
