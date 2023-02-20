/**
 * @file io.hpp
 * PDI output functions
 */
#pragma once

#include <array>
#include "grid.hpp"

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out);

void write_pdi_init(int max_iter, int frequency, Grid *grid);

void write_pdi(int iter, double t, void * rho, void *u, void *P);
