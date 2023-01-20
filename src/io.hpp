/**
 * @file io.hpp
 * PDI output functions
 */
#pragma once

void init_write(int max_iter, int frequency, int ghost);

void write(int iter, int nx, void* rho, void * u);
