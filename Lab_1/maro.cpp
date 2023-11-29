#include "maro.h"

std::vector<profiling_results_t> run_experiement_omp(double(*f)(const double*, size_t), size_t N, std::unique_ptr<double[]>& arr) {
	std::vector<profiling_results_t> res_table;
	unsigned threads_count = omp_get_num_procs();

	for (unsigned T = 1; T <= threads_count; ++T) {
		res_table.emplace_back();

		omp_set_num_threads(T);
		res_table[T - 1].T = T;

		auto t1 = omp_get_wtime();
		res_table[T - 1].result = f(arr.get(), N);
		auto t2 = omp_get_wtime();

		res_table[T - 1].time = t2 - t1;
		res_table[T - 1].speedup = res_table[0].time / res_table[T - 1].time;
		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
	}

	return res_table;
}

std::vector<profiling_results_t> run_experiement_cpp(double (*f)(const double*, size_t), size_t N, std::unique_ptr<double[]>& arr) {
	using namespace std::chrono;

	std::vector<profiling_results_t> res_table;
	std::size_t T_max = get_num_threads();

	for (unsigned T = 1; T <= T_max; ++T) {
		set_num_threads(T);

		res_table.emplace_back();
		res_table[T - 1].T = T;

		auto t1 = steady_clock::now();
		res_table[T - 1].result = f(arr.get(), N);
		auto t2 = steady_clock::now();

		res_table[T - 1].time = duration_cast<milliseconds>(t2 - t1).count();
		res_table[T - 1].speedup = res_table[0].time / res_table[T - 1].time;
		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
	}

	return res_table;
}