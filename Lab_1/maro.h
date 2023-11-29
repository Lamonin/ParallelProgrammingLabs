#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <omp.h>
#include "thread_misc.h"

struct profiling_results_t {
	double result, time, speedup, efficiency;
	unsigned T;
};

std::vector<profiling_results_t> run_experiement_omp(double(*f)(const double*, size_t), size_t N, std::unique_ptr<double[]>& arr);
std::vector<profiling_results_t> run_experiement_cpp(double (*f)(const double*, size_t), size_t N, std::unique_ptr<double[]>& arr);