/**
 * @file io.hpp
 * PDI output functions
 */
#pragma once

void init_write(int max_iter, int frequency);

void write(int iter, int nx, void* rho);
