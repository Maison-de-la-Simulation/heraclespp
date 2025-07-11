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

                // 20 clumps until 4.65e11 m, 15 between the two shocks
                // v1d 9
                /* Kokkos::Array<double, 20> r_pos_clump = {3.43452771e+11, 1.77298726e+11, 3.40513417e+11, 1.36647263e+11,
                    1.65819797e+11, 4.50461474e+11, 3.85707678e+11, 4.06880918e+11,
                    4.32887613e+11, 3.63889951e+11, 4.61718550e+11, 3.75922642e+11,
                    4.39919189e+11, 4.40020104e+11, 3.95555551e+11, 3.63888013e+11,
                    4.18960379e+11, 3.73747159e+11, 3.75281290e+11, 4.55100978e+11};

                //m 25
                Kokkos::Array<double, 20> r_pos_clump = {2.69826033e+11, 1.76488326e+11, 1.89893639e+11, 2.11275630e+11,
                    2.41878303e+11, 4.18551860e+11, 4.22626892e+11, 4.35829408e+11,
                    3.85561020e+11, 3.70830581e+11, 3.58815903e+11, 4.55656646e+11,
                    3.70665294e+11, 4.40900614e+11, 4.03392645e+11, 3.96782309e+11,
                    4.58587589e+11, 3.53371787e+11, 4.21489361e+11, 3.92864232e+11}; */

                // 20 clumps until sous la couche H
                // v1d 9
                Kokkos::Array<double, 20> r_pos_clump = { 1.89961235e+11, 2.04238098e+11, 2.17861250e+11, 2.66025000e+11,
                    2.77253313e+11, 3.00259129e+11, 3.03851454e+11, 3.18892260e+11,
                    3.52418492e+11, 3.54230286e+11, 3.55675654e+11, 3.60100509e+11,
                    3.61254211e+11, 3.63136017e+11, 3.64369659e+11, 3.64729247e+11,
                    3.65375314e+11, 3.68025069e+11, 3.68662201e+11, 3.70051478e+11};

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
                // v1d 9
                Kokkos::Array<double, 20> rad_clump = {2.7588201320643764, 2.661550941810921, 2.621417676977413,
                    2.6186418881256968, 2.539790682241017, 2.437366067905366, 2.39295048451544, 2.358231540377209,
                    2.1205412257092675, 2.0765138209455136, 2.0350537847928845, 2.0198651992646006, 1.9745075486172643, 1.8918290026628628, 1.8865384081224315,
                    1.8552655834012088, 1.847326765037701, 1.7954394634338837, 1.788854081486872, 1.7472145148853444};
                //m 25
                /* Kokkos::Array<double, 20> rad_clump = {1.9962232962360296, 1.9830574320162917, 1.9697672919295248,
                    1.9644156273275757, 1.9388850981598487, 1.9364569406571923, 1.9077424194415376, 1.8939470546902113,
                    1.8714536595713858, 1.8661661494060326, 1.8512553921277288, 1.8445007874495445, 1.8423743836896884,
                    1.8333416693148512, 1.8151809116663735, 1.7903097694318548, 1.765960365083364, 1.7639321965089676,
                    1.7613133757887607, 1.7176733340042862}; */

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
