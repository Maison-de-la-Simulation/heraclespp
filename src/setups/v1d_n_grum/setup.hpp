// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

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
    double pos_ni_bubble_1;
    double pos_ni_bubble_2;
    double pos_ni_bubble_3;
    double pos_ni_bubble_4;
    double pos_ni_bubble_5;
    double pos_ni_bubble_6;
    double pos_ni_bubble_7;
    double pos_ni_bubble_8;
    double pos_ni_bubble_9;
    double pos_ni_bubble_10;
    double pos_ni_bubble_11;
    double pos_ni_bubble_12;
    double pos_ni_bubble_13;
    double pos_ni_bubble_14;
    double pos_ni_bubble_15;
    double pos_ni_bubble_16;
    double pos_ni_bubble_17;
    double pos_ni_bubble_18;
    double pos_ni_bubble_19;
    double pos_ni_bubble_20;
    double rad_clump_1;
    double rad_clump_2;
    double rad_clump_3;
    double rad_clump_4;
    double rad_clump_5;
    double rad_clump_6;
    double rad_clump_7;
    double rad_clump_8;
    double rad_clump_9;
    double rad_clump_10;
    double rad_clump_11;
    double rad_clump_12;
    double rad_clump_13;
    double rad_clump_14;
    double rad_clump_15;
    double rad_clump_16;
    double rad_clump_17;
    double rad_clump_18;
    double rad_clump_19;
    double rad_clump_20;

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
        pos_ni_bubble_1 = reader.GetReal("Initialisation", "pos_ni_bubble_1", 0.);
        pos_ni_bubble_2 = reader.GetReal("Initialisation", "pos_ni_bubble_2", 0.);
        pos_ni_bubble_3 = reader.GetReal("Initialisation", "pos_ni_bubble_3", 0.);
        pos_ni_bubble_4 = reader.GetReal("Initialisation", "pos_ni_bubble_4", 0.);
        pos_ni_bubble_5 = reader.GetReal("Initialisation", "pos_ni_bubble_5", 0.);
        pos_ni_bubble_6 = reader.GetReal("Initialisation", "pos_ni_bubble_6", 0.);
        pos_ni_bubble_7 = reader.GetReal("Initialisation", "pos_ni_bubble_7", 0.);
        pos_ni_bubble_8 = reader.GetReal("Initialisation", "pos_ni_bubble_8", 0.);
        pos_ni_bubble_9 = reader.GetReal("Initialisation", "pos_ni_bubble_9", 0.);
        pos_ni_bubble_10 = reader.GetReal("Initialisation", "pos_ni_bubble_10", 0.);
        pos_ni_bubble_11 = reader.GetReal("Initialisation", "pos_ni_bubble_11", 0.);
        pos_ni_bubble_12 = reader.GetReal("Initialisation", "pos_ni_bubble_12", 0.);
        pos_ni_bubble_13 = reader.GetReal("Initialisation", "pos_ni_bubble_13", 0.);
        pos_ni_bubble_14 = reader.GetReal("Initialisation", "pos_ni_bubble_14", 0.);
        pos_ni_bubble_15 = reader.GetReal("Initialisation", "pos_ni_bubble_15", 0.);
        pos_ni_bubble_16 = reader.GetReal("Initialisation", "pos_ni_bubble_16", 0.);
        pos_ni_bubble_17 = reader.GetReal("Initialisation", "pos_ni_bubble_17", 0.);
        pos_ni_bubble_18 = reader.GetReal("Initialisation", "pos_ni_bubble_18", 0.);
        pos_ni_bubble_19 = reader.GetReal("Initialisation", "pos_ni_bubble_19", 0.);
        pos_ni_bubble_20 = reader.GetReal("Initialisation", "pos_ni_bubble_20", 0.);
        rad_clump_1  = reader.GetReal("Initialisation", "rad_clump_1", 0.);
        rad_clump_2  = reader.GetReal("Initialisation", "rad_clump_2", 0.);
        rad_clump_3  = reader.GetReal("Initialisation", "rad_clump_3", 0.);
        rad_clump_4  = reader.GetReal("Initialisation", "rad_clump_4", 0.);
        rad_clump_5  = reader.GetReal("Initialisation", "rad_clump_5", 0.);
        rad_clump_6  = reader.GetReal("Initialisation", "rad_clump_6", 0.);
        rad_clump_7  = reader.GetReal("Initialisation", "rad_clump_7", 0.);
        rad_clump_8  = reader.GetReal("Initialisation", "rad_clump_8", 0.);
        rad_clump_9  = reader.GetReal("Initialisation", "rad_clump_9", 0.);
        rad_clump_10 = reader.GetReal("Initialisation", "rad_clump_10", 0.);
        rad_clump_11 = reader.GetReal("Initialisation", "rad_clump_11", 0.);
        rad_clump_12 = reader.GetReal("Initialisation", "rad_clump_12", 0.);
        rad_clump_13 = reader.GetReal("Initialisation", "rad_clump_13", 0.);
        rad_clump_14 = reader.GetReal("Initialisation", "rad_clump_14", 0.);
        rad_clump_15 = reader.GetReal("Initialisation", "rad_clump_15", 0.);
        rad_clump_16 = reader.GetReal("Initialisation", "rad_clump_16", 0.);
        rad_clump_17 = reader.GetReal("Initialisation", "rad_clump_17", 0.);
        rad_clump_18 = reader.GetReal("Initialisation", "rad_clump_18", 0.);
        rad_clump_19 = reader.GetReal("Initialisation", "rad_clump_19", 0.);
        rad_clump_20 = reader.GetReal("Initialisation", "rad_clump_20", 0.);
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
        auto const r = grid.x;
        auto const dx = grid.dx;
        auto const dv = grid.dv;
        auto const& param_setup = m_param_setup;

        if (ndim == 3)
        {
            // add Ni bubble 3D
            auto const th = grid.y;
            auto const phi = grid.z;
            auto const param_setup = m_param_setup;
            double const th_mid = (param_setup.angle_min + param_setup.angle_max) / 2;
            double const phi_mid = (param_setup.angle_min + param_setup.angle_max) / 2;

            Kokkos::parallel_for(
                "Ni_clump_3D",
                cell_mdrange(range),
                KOKKOS_LAMBDA(int i, int j, int k)
                {
                fx(i, j, k, 0) = 0;
                double x_cart = r(i) * Kokkos::sin(th(j)) * Kokkos::cos(phi(k));
                double y_cart = r(i) * Kokkos::sin(th(j)) * Kokkos::sin(phi(k));
                double z_cart = r(i) * Kokkos::cos(th(j));

                Kokkos::Array<double, 20> r_pos_clump = {param_setup.pos_ni_bubble_1,
                    param_setup.pos_ni_bubble_2, param_setup.pos_ni_bubble_3, param_setup.pos_ni_bubble_4,
                    param_setup.pos_ni_bubble_5, param_setup.pos_ni_bubble_6, param_setup.pos_ni_bubble_7,
                    param_setup.pos_ni_bubble_8, param_setup.pos_ni_bubble_9, param_setup.pos_ni_bubble_10,
                    param_setup.pos_ni_bubble_11, param_setup.pos_ni_bubble_12, param_setup.pos_ni_bubble_13,
                    param_setup.pos_ni_bubble_14, param_setup.pos_ni_bubble_15, param_setup.pos_ni_bubble_16,
                    param_setup.pos_ni_bubble_17, param_setup.pos_ni_bubble_18, param_setup.pos_ni_bubble_19,
                    param_setup.pos_ni_bubble_20
                };

                Kokkos::Array<double, 20> rad_clump = {param_setup.rad_clump_1,
                    param_setup.rad_clump_2, param_setup.rad_clump_3, param_setup.rad_clump_4,
                    param_setup.rad_clump_5, param_setup.rad_clump_6, param_setup.rad_clump_7,
                    param_setup.rad_clump_8, param_setup.rad_clump_9, param_setup.rad_clump_10,
                    param_setup.rad_clump_11, param_setup.rad_clump_12, param_setup.rad_clump_13,
                    param_setup.rad_clump_14, param_setup.rad_clump_15, param_setup.rad_clump_16,
                    param_setup.rad_clump_17, param_setup.rad_clump_18, param_setup.rad_clump_19,
                    param_setup.rad_clump_20
                };

                /*

                // m25
                Kokkos::Array<double, 20> r_pos_clump = {1.77141887e+11, 1.78791637e+11, 1.91855706e+11, 2.11104949e+11,
                    2.52245467e+11, 2.62345276e+11, 3.05792752e+11, 3.16184701e+11,
                    3.25017366e+11, 3.41402412e+11, 3.50561781e+11, 3.52168683e+11,
                    3.52253582e+11, 3.53098437e+11, 3.53227268e+11, 3.54037414e+11,
                    3.54555585e+11, 3.55001788e+11, 3.55529775e+11, 3.56048270e+11}; */

                Kokkos::Array<double, 20> th_pos_clump = {1.57062107, 1.23285187, 1.70336447,
                    1.84833712, 1.2581159, 1.37732577,
                    1.27876188, 1.45761171, 1.80546387, 1.78652657, 1.76958736, 1.85559125,
                    1.61251361, 1.76122944, 1.26368143, 1.83131898, 1.32587535, 1.46288975,
                    1.69304861, 1.6208343};

                Kokkos::Array<double, 20> phi_pos_clump = {1.30631996, 1.75716046, 1.52559179, 1.41205752, 1.91193411, 1.5333103,
                    1.70334415, 1.36324587, 1.8904564, 1.77684056, 1.855729, 1.42458767,
                    1.84821715, 1.2027297, 1.39796724, 1.317004, 1.52696258, 1.24268949,
                    1.69077599, 1.69659713};

                //double rad_clump = 3.5E10;
                /*
                //m 25
                Kokkos::Array<double, 20> rad_clump = {1.579066911049704, 1.5783825538889114, 1.57383143975951,
                    1.5153073261201304, 1.5032456903104354, 1.4901662073686701, 1.4763414254507166, 1.4760081523806399,
                    1.4342193824328078, 1.4142044388513932, 1.4042037432917414, 1.3958618324536198, 1.3709117574558876,
                    1.3144371813337081, 1.2974845282661625, 1.2669518831053048, 1.2435433963418836, 1.232523742752487,
                    1.2300949264578374, 1.1528002795360854}; */

                for (int iclump = 0; iclump < 20; ++iclump)
                {
                    double x_center = r_pos_clump[iclump] * Kokkos::sin(th_pos_clump[iclump]) * Kokkos::cos(phi_pos_clump[iclump]);
                    double y_center = r_pos_clump[iclump] * Kokkos::sin(th_pos_clump[iclump]) * Kokkos::sin(phi_pos_clump[iclump]);
                    double z_center = r_pos_clump[iclump] * Kokkos::cos(th_pos_clump[iclump]);

                    double dist = Kokkos::sqrt((x_cart - x_center)*(x_cart - x_center)
                            + (y_cart - y_center)*(y_cart - y_center)
                            + (z_cart - z_center)*(z_cart - z_center));

                    if (dist <= rad_clump[iclump] * 1E10)
                    //if (dist <= rad_clump)
                    {
                        fx(i, j, k, 0) = 1;
                        fx(i, j, k, 1) = 0;
                        fx(i, j, k, 2) = 0;
                        fx(i, j, k, 3) = 0;
                        fx(i, j, k, 4) = 0;
                    }
                }
            });
        }

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
                //fx(i, j, k, 1) = param_setup.Other_CL;
                fx(i, j, k, 1) = param_setup.H_CL;
                fx(i, j, k, 2) = param_setup.He_CL;
                fx(i, j, k, 3) = param_setup.O_CL;
                fx(i, j, k, 4) = param_setup.Si_CL;
            });
    }
};

} // namespace novapp
