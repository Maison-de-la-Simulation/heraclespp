// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>
#include <Kokkos_Random.hpp>
#include <numeric>

#include <inih/INIReader.hpp>

#include "broadcast.hpp"
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
#include <shift_criterion_interface.hpp>
#include <hdf5.h>
#include <io_hdf5.hpp>

namespace novapp
{

class ParamSetup
{
public:
    std::string init_filename;
    double vmax_shift;
    int cell_shift;
    double angle_min;
    double angle_max;
    double v0_CL;
    double R_star;
    double T_CL;
    double Ni_CL;
    double H_CL;
    double He_CL;
    double O_CL;
    double Si_CL;
    double Other_CL;
    double pos_ni_bubble;
    double radius_ni_bubble;
    double pos_ni_bubble_2;
    double radius_ni_bubble_2;
    double pos_ni_bubble_3;
    double radius_ni_bubble_3;

    explicit ParamSetup(INIReader const& reader)
    {
        init_filename = reader.Get("Problem", "init_file", "");
        vmax_shift = reader.GetReal("Problem", "vmax_shift", 0.);
        cell_shift = reader.GetInteger("Problem", "cell_shift", 0);
        angle_min = reader.GetReal("Grid", "ymin", 1.0);
        angle_max = reader.GetReal("Grid", "ymax", 1.0);
        v0_CL = reader.GetReal("Boundary Condition", "v0_CL", 0.);
        R_star = reader.GetReal("Boundary Condition", "R_star", 0.);
        T_CL = reader.GetReal("Boundary Condition", "T_CL", 0.);
        Ni_CL = reader.GetReal("Boundary Condition", "Ni_CL", 0.);
        H_CL = reader.GetReal("Boundary Condition", "H_CL", 0.);
        He_CL = reader.GetReal("Boundary Condition", "He_CL", 0.);
        O_CL = reader.GetReal("Boundary Condition", "O_CL", 0.);
        Si_CL = reader.GetReal("Boundary Condition", "Si_CL", 0.);
        Other_CL = reader.GetReal("Boundary Condition", "Other_CL", 0.);
        pos_ni_bubble = reader.GetReal("Initialisation", "pos_ni_bubble", 0.);
        radius_ni_bubble = reader.GetReal("Initialisation", "radius_ni_bubble", 0.);
        pos_ni_bubble_2 = reader.GetReal("Initialisation", "pos_ni_bubble_2", 0.);
        radius_ni_bubble_2 = reader.GetReal("Initialisation", "radius_ni_bubble_2", 0.);
        pos_ni_bubble_3 = reader.GetReal("Initialisation", "pos_ni_bubble_3", 0.);
        radius_ni_bubble_3 = reader.GetReal("Initialisation", "radius_ni_bubble_3", 0.);
   }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    EOS m_eos;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        EOS const& eos,
        ParamSetup param_set_up,
        [[maybe_unused]] Gravity const& gravity)
        : m_eos(eos)
        , m_param_setup(std::move(param_set_up))
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);

        KDV_double_1d rho_1d("rho_1d", grid.Nx_local_ng[0]);
        KDV_double_1d u_1d("u_1d", grid.Nx_local_ng[0]);
        KDV_double_1d P_1d("P_1d", grid.Nx_local_ng[0]);
        KDV_double_2d fx_1d("fx_1d", grid.Nx_local_ng[0], fx.extent_int(3));

        std::size_t const extent = grid.Nx_glob_ng[0];

        raii_h5_hid const file_id(::H5Fopen(m_param_setup.init_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), ::H5Fclose);
        check_extent_dset(file_id, "/rho_1d", std::array {extent});
        check_extent_dset(file_id, "/u_1d", std::array {extent});
        check_extent_dset(file_id, "/P_1d", std::array {extent});
        check_extent_dset(file_id, "/fx_1d", std::array {fx_1d.extent(1), extent});

        if (grid.mpi_rank == 0)
        {
            std::cout << std::setw(81) << std::setfill('*') << '\n';
            std::cout << "reading from file " << m_param_setup.init_filename << '\n';
        }

        int const filename_size = m_param_setup.init_filename.size();
        // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        PDI_multi_expose(
            "read_hydro_1d",
            "nullptr", nullptr, PDI_OUT,
            "init_filename_size", &filename_size, PDI_OUT,
            "init_filename", m_param_setup.init_filename.data(), PDI_OUT,
            "rho_1d", rho_1d.view_host().data(), PDI_INOUT,
            "u_1d", u_1d.view_host().data(), PDI_INOUT,
            "P_1d", P_1d.view_host().data(), PDI_INOUT,
            "fx_1d", fx_1d.view_host().data(), PDI_INOUT,
            nullptr);
        // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        modify_host(rho_1d, u_1d, P_1d, fx_1d);
        sync_device(rho_1d, u_1d, P_1d, fx_1d);

        broadcast(range, rho_1d.view_device(), rho);
        broadcast(range, u_1d.view_device(), Kokkos::subview(u, ALL, ALL, ALL, 0));
        for (int idim = 1; idim < u.extent_int(3); ++idim)
        {
            broadcast(range, 0, Kokkos::subview(u, ALL, ALL, ALL, idim));
        }
        broadcast(range, P_1d.view_device(), P);
        for(int ifx = 0; ifx < fx.extent_int(3); ++ifx)
        {
            broadcast(range, Kokkos::subview(fx_1d.view_device(), ALL, ifx), Kokkos::subview(fx, ALL, ALL, ALL, ifx));
        }

        // clump au centre + perturbation
        auto const rc = grid.x_center;
        auto const r = grid.x;
        auto const thc = grid.y_center;
        auto const phic = grid.z_center;
        auto const dv = grid.dv;

        Kokkos::Random_XorShift64_Pool<> random_pool(54321);

        Kokkos::parallel_for(
            "Ni_clump_3D",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double r_max_ni = 1.245E9;
                int n_mode = 500;
                int m_mode = 800;
                double perturbed_r_max = r_max_ni * (1.0 + 0.01 * Kokkos::sin(n_mode * thc(j)) * Kokkos::cos(m_mode * phic(k)));

                double noise = 0;
                auto generator = random_pool.get_state();
                noise = generator.drand(-1.0, 1.0);
                random_pool.free_state(generator);
                double perturbed_r_max2 = r_max_ni * (1.0 + 0.01 * noise * noise);

                if (rc(i) <= perturbed_r_max2)
                {
                    fx(i, j, k, 0) = 1;
                    fx(i, j, k, 1) = 0;
                }
            });

        /* if (ndim == 3)
        {
            // clump au centre + perturbation
            auto const th = grid.y;
            auto const phi = grid.z;
            auto const thc = grid.y_center;
            auto const phic = grid.z_center;
            auto const param_setup = m_param_setup;
            double const th_mid = (param_setup.angle_min + param_setup.angle_max) / 2;
            double const phi_mid = (param_setup.angle_min + param_setup.angle_max) / 2;

            int const nx = rho.extent_int(0) - 2 * grid.Nghost[0];
            int const ny = rho.extent_int(1) - 2 * grid.Nghost[1];
            int const nz = rho.extent_int(2) - 2 * grid.Nghost[2];
            int const ng_x = grid.Nghost[0];
            Kokkos::Random_XorShift64_Pool<> random_pool(54321);
            int kx1 = 60;
            int kx2 = 80;

            Kokkos::Array<int, 5> kx;
            std::iota(kx.data(), kx.data() + 5, 0);
            Kokkos::Array<int, 5> ky;
            std::iota(ky.data(), ky.data() + 5, 0);

            Kokkos::parallel_for(
                "Ni_clump_3D",
                cell_mdrange(range),
                KOKKOS_LAMBDA(int i, int j, int k)
                {
                    fx(i, j, k, 0) = 0;

                    // perturbation en rho
                    double petrurb_r = 0;
                    double perturb_th_ph = 0;
                    double noise = 0;

                    // background noise
                    auto generator = random_pool.get_state();
                    noise = generator.drand(-1.0, 1.0);

                    double A = 0.015;
                    for (int n = 0; n < 10; ++n)
                    {
                        double kth = ny / (generator.drand(0.0, 1.0) * 50);
                        double kph = nz / (generator.drand(0.0, 1.0) * 50);
                        double alpha1 = (ny + nz) * 10 * generator.drand(0.0, 1.0);
                        double alpha2 = (ny + nz) * 10 * generator.drand(0.0, 1.0);

                        perturb_th_ph += A * Kokkos::sin(j / kth + alpha1) * Kokkos::cos(k / kph + alpha2);
                    }

                    double rmin = r(ng_x);
                    double rmax = r(nx);
                    double L = rmax - rmin;
                    double sigma = 0.1 * L * L;
                    petrurb_r = Kokkos::exp(-(rc(i)) * (rc(i)) / sigma) * Kokkos::cos(kx1 * rc(i) + kx2 * rc(i));

                    random_pool.free_state(generator);

                    double perturb = 0.1 * petrurb_r + perturb_th_ph;
                    double rho_av = rho(i, j, k);
                    rho(i, j, k) = rho(i, j, k) * (1 + perturb + 0.01 * noise);

                    // perturbation interface fx
                    double X = rc(i) * sin(thc(j)) * cos(phic(k));
                    double Y = rc(i) * sin(thc(j)) * sin(phic(k));
                    double h = 0;
                    double ak = 0.861466;
                    double bk = 0.940694;
                    double ck = 0.786941;
                    double dk = 0.565028;

                    for (std::size_t ik = 0; ik < Kokkos::Array<int, 5>::size(); ++ik)
                    {
                        for (std::size_t jk = 0; jk < Kokkos::Array<int, 5>::size(); ++jk)
                        {
                            double const K = kx[ik] * kx[ik] + ky[jk] * ky[jk];
                            if (K >= 8 && K <= 16)
                            {
                                h += (ak * Kokkos::cos(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                                    + bk * Kokkos::cos(kx[ik] * X) * Kokkos::sin(ky[jk] * Y)
                                    + ck * Kokkos::sin(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                                    + dk * Kokkos::sin(kx[ik] * X) * Kokkos::sin(ky[jk] * Y));
                            }
                        }
                    }
                    double perturbed_r_max2 = r_max_ni * (1 + 0.01 * h * h);

                    double r_max_ni = 2E11;
                    int n_mode = 900;
                    int m_mode = 800;
                    double perturbed_r_max = r_max_ni * (1.0 + 0.01 * Kokkos::sin(n_mode * th(j)) * Kokkos::cos(m_mode * phi(k)));

                    if (r(i) <= perturbed_r_max)
                    {
                        fx(i, j, k, 0) = 1;
                        fx(i, j, k, 1) = 0;
                    }
            });
        } */

        double sum_M_ni = 0;

        Kokkos::parallel_reduce(
        "sum_mass_Ni",
        cell_mdrange(range),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& local_sum)
        {
            local_sum += fx(i, j, k, 0) * rho(i, j, k) * dv(i, j, k);
        },
        Kokkos::Sum<double>(sum_M_ni));

        MPI_Allreduce(MPI_IN_PLACE, &sum_M_ni, 1, MPI_DOUBLE, MPI_SUM, grid.comm_cart);

        if (grid.mpi_rank == 0)
        {
            Kokkos::printf("Masse Ni56 = %f M_sun\n", sum_M_ni / (1.9891E30));
        }
    }
};

class GridSetup : public IGridType
{
private :
    Param m_param;

public:
    explicit GridSetup(Param param)
        : m_param(std::move(param))
    {
    }

    void execute(
        std::array<int, 3> const Nghost,
        std::array<int, 3> const Nx_glob_ng,
        KVH_double_1d const& x_glob,
        KVH_double_1d const& y_glob,
        KVH_double_1d const& z_glob) const final
    {
        std::string const init_file = m_param.reader.Get("problem", "init_file", "");

        raii_h5_hid const file_id(::H5Fopen(init_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), ::H5Fclose);
        check_extent_dset(file_id, "/x", std::array {x_glob.extent(0)});

        int const filename_size = init_file.size();
        // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        PDI_multi_expose(
            "read_mesh_1d",
            "nullptr", nullptr, PDI_OUT,
            "init_filename_size", &filename_size, PDI_OUT,
            "init_filename", init_file.data(), PDI_OUT,
            "x", x_glob.data(), PDI_INOUT,
            nullptr);
        // NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)

        compute_regular_mesh_1d(y_glob, Nghost[1], m_param.ymin, (m_param.ymax - m_param.ymin) / Nx_glob_ng[1]);
        compute_regular_mesh_1d(z_glob, Nghost[2], m_param.zmin, (m_param.zmax - m_param.zmin) / Nx_glob_ng[2]);
    }
};

class UserShiftCriterion : public IShiftCriterion
{
private:
    ParamSetup m_param_setup;

public:
    explicit UserShiftCriterion(ParamSetup param_setup)
        : m_param_setup(std::move(param_setup))
    {
    }

    [[nodiscard]] bool execute(
        [[maybe_unused]] Range const& range,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& rhou,
        [[maybe_unused]] KV_double_3d const& E,
        [[maybe_unused]] KV_double_4d const& fx) const override
    {
        bool exit_bool = grid.Nx_local_ng[0] < m_param_setup.cell_shift;
        std::array const root_coords {grid.mpi_dims_cart[0] - 1, 0, 0};
        int root = -1;
        MPI_Cart_rank(grid.comm_cart, root_coords.data(), &root);
        MPI_Bcast(&exit_bool, 1, MPI_CXX_BOOL, root, grid.comm_cart);

        if (exit_bool)
        {
            throw std::runtime_error("The shift criterion is greater than the size of the last MPI process");
        }

        double vmax = 0;

        if (grid.mpi_rank_cart[0] == grid.mpi_dims_cart[0] - 1)
        {
            int const ishift_min = grid.Nx_local_ng[0] - m_param_setup.cell_shift;
            Kokkos::parallel_reduce(
                "shift criterion",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {ishift_min, grid.Nghost[1], grid.Nghost[2]},
                {grid.Nghost[0] + grid.Nx_local_ng[0],
                grid.Nghost[1] + grid.Nx_local_ng[1],
                grid.Nghost[2] + grid.Nx_local_ng[2]}),
                KOKKOS_LAMBDA(int i, int j, int k, double& vloc)
                {
                    vloc = Kokkos::max(rhou(i, j, k, 0) / rho(i, j, k), vloc);
                },
                Kokkos::Max<double>(vmax));
        }
        MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_DOUBLE, MPI_MAX, grid.comm_cart);
        return vmax >= m_param_setup.vmax_shift;
    }
};

template <class Gravity>
class BoundarySetup : public IBoundaryCondition<Gravity>
{
    std::string m_label;

private:
    EOS m_eos;
    ParamSetup m_param_setup;

public:
    BoundarySetup(int idim, int iface,
        EOS const& eos,
        ParamSetup const& param_setup)
        : IBoundaryCondition<Gravity>(idim, iface)
        , m_label(std::string("UserDefined").append(bc_dir(idim)).append(bc_face(iface)))
        , m_eos(eos)
        , m_param_setup(param_setup)
    {
    }

    void execute(Grid const& grid,
                 [[maybe_unused]] Gravity const& gravity,
                 KV_double_3d const& rho,
                 KV_double_4d const& rhou,
                 KV_double_3d const& E,
                 KV_double_4d const& fx) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        auto const xc = grid.x_center;
        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;
        double const mu = m_eos.mean_molecular_weight();

        int const ng = grid.Nghost[this->bc_idim()];
        if (this->bc_iface() == 1)
        {
            begin[this->bc_idim()] = rho.extent_int(this->bc_idim()) - ng;
        }
        end[this->bc_idim()] = begin[this->bc_idim()] + ng;

        Kokkos::parallel_for(
            m_label,
            Kokkos::MDRangePolicy<int, Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double u_rgs = 50000; // m s^{-1}
                double M_sun = 1.989e30; // kg
                double m_dot_rsg = 1e-6 * M_sun / (365 * 24 * 3600); // kg s^{-1}

                double u_r = param_setup.v0_CL + (u_rgs - param_setup.v0_CL)
                            * (1 - param_setup.R_star / xc(i)) * (1 - param_setup.R_star / xc(i));

                rho(i, j, k) = m_dot_rsg / (4 * units::pi * xc(i) * xc(i) * u_r);

                rhou(i, j, k, 0) = rho(i, j, k) * u_r;

                if (ndim == 2 || ndim == 3)
                {
                    rhou(i, j, k, 1) = 0;
                }
                if (ndim == 3)
                {
                    rhou(i, j, k, 2) = 0;
                }

                E(i, j, k) = eos.compute_evol_from_T(rho(i, j, k), param_setup.T_CL);

                fx(i, j, k, 0) = param_setup.Ni_CL;
                fx(i, j, k, 1) = param_setup.Other_CL;
                /* fx(i, j, k, 1) = param_setup.H_CL;
                fx(i, j, k, 2) = param_setup.He_CL;
                fx(i, j, k, 3) = param_setup.O_CL;
                fx(i, j, k, 4) = param_setup.Si_CL; */
            });
    }
};

} // namespace novapp
