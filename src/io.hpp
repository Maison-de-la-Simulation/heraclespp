/**
 * @file io.hpp
 * PDI output functions
 */
#pragma once

# include <array>

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out);

void init_write(int max_iter, int frequency, std::array<int,3> Nghost);

void write(int iter, int* nx, double current, void * rho, void *u, void *P);

