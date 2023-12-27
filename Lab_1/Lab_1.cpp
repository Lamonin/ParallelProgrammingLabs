#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <queue>
#include <omp.h>
#include <mutex>
#include <io.h>
#include <functional>
#include "maro.h"
#include "thread_misc.h"
#include <locale>
#include <stdio.h>

using namespace std;


double average(const uint32_t* V, size_t n) {
	double res = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res += V[i];
	}
	return res / n;
}

double average_reduce(const uint32_t* V, size_t n) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int i = 0; i < n; ++i) {
		res += V[i];
	}
	return res / n;
}

double average_rr(const uint32_t* V, size_t n) {
	double res = 0.0;

#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_max_threads();
		for (int i = t; i < n; i += T) {
			res += V[i]; // Гонка потоков (на этом кончилась пара)
		}
	}

	return res / n;
}

double average_omp(const uint32_t* V, size_t n) {
	// double res = 0.0, * partial_sums = (double*)calloc(omp_get_num_procs(), sizeof(double));
	double res = 0.0, * partial_sums;

#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_max_threads();
#pragma omp single
		{
			partial_sums = (double*)malloc(T * sizeof(V[0]));
		}
		partial_sums[t] = 0.0;
		for (int i = t; i < n; i += T) {
			partial_sums[t] += V[i];
		}
	}
	for (size_t i = 1; i < omp_get_num_procs(); ++i) {
		partial_sums[0] += partial_sums[i];
	}
	res = partial_sums[0] / n;
	free(partial_sums);
	return res;
}

struct partial_sum_t {
	/*union {
		double value;
		char padd[64];
	};*/
	alignas (64) double value = 0.0;
};

double average_omp_aligned(const uint32_t* V, size_t n) {
	unsigned T;
	double res = 0.0;
	partial_sum_t* partial_sums;

#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_max_threads();
#pragma omp single
		{
			partial_sums = (partial_sum_t*)malloc(T * sizeof(partial_sum_t));
		}
		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) {
			partial_sums[t].value += V[i];
		}
	}
	for (size_t i = 1; i < T; ++i) {
		partial_sums[0].value += partial_sums[i].value;
	}
	res = partial_sums[0].value / n;
	free(partial_sums);
	return res;
}

double average_cpp_aligned(const uint32_t* V, size_t n) {
	unsigned T;
	double res = 0.0;
	std::unique_ptr<partial_sum_t[]> partial_sums;
#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_max_threads();
#pragma omp single
		{
			partial_sums = std::make_unique<partial_sum_t[]>(T);
		}

		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) {
			partial_sums[t].value += V[i];
		}
	}
	for (size_t i = 1; i < T; ++i) {
		partial_sums[0].value += partial_sums[i].value;
	}
	return partial_sums[0].value / n;
}

double average_omp_mtx(const uint32_t* V, size_t n) {
	double res = 0.0;
#pragma omp parallel
	{
		double partial_sum = 0.0;
		unsigned T = omp_get_num_threads();
		unsigned t = omp_get_thread_num();
		for (int i = t; i < n; i += T)
			partial_sum += V[i];
#pragma omp critical
		{
			res += partial_sum;
		}
	}

	return res / n;
}

double average_cpp_mtx(const uint32_t* V, size_t N) {
	double res = 0.0;
	unsigned T = get_num_threads();
	std::vector<std::thread> workers;
	std::mutex mtx;

	auto worker_proc = [&mtx, T, V, N, &res](unsigned t) {
		double partial_sum = 0.0;
		for (size_t i = t; i < N; i += T) {
			partial_sum += V[i];
		}
		{
			//Start from C++ 17
			std::scoped_lock l{ mtx };
			res += partial_sum;
		}
		// OR
		//mtx.lock();
		//res += partial_sum;
		//mtx.unlock();
		};

	for (unsigned t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}
	worker_proc(0);
	for (auto& w : workers) {
		w.join();
	}

	return res / N;
}

// Локализация 18.10.2023
double average_cpp_mtx_local(const uint32_t* V, size_t N) {
	double res = 0.0;
	unsigned T = get_num_threads();
	std::vector<std::thread> workers;
	std::mutex mtx;

	auto worker_proc = [&mtx, &res, T, V, N](unsigned t) {
		size_t b = N % T, e = N / T;
		if (t < b)
			b = t * ++e;
		else
			b += t * e;
		e += b;

		double partial_sum = 0.0;
		for (int i = b; i < e; ++i) {
			partial_sum += V[i];
		}

		mtx.lock();
		res += partial_sum;
		mtx.unlock();
		};

	for (unsigned t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}

	worker_proc(0);
	for (auto& w : workers) {
		w.join();
	}

	return res / N;
}

double average_cpp_reduction(const uint32_t* V, size_t N) {
	unsigned T = get_num_threads();

	std::vector<double> partial_sums;
	partial_sums.resize(T);

	std::vector<std::thread> workers;

	barrier	bar(T);

	auto worker_proc = [&bar, &partial_sums, T, &V, N](unsigned t) {
		size_t b = N % T, e = N / T;

		if (t < b)
			b = t * ++e;
		else
			b += t * e;
		e += b;

		double partial_sum = 0.0;
		for (int i = b; i < e; ++i) {
			partial_sum += V[i];
		}

		partial_sums[t] = partial_sum;

		for (std::size_t step = 1, next = 2; step < T; step = next, next += next) {
			bar.arrive_and_wait();
			if ((t & (next - 1)) == 0 && t + step < T) {  // t % next
				partial_sums[t] += partial_sums[t + step];
			}
		}
		};

	for (unsigned t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}

	worker_proc(0);
	for (auto& w : workers) {
		w.join();
	}

	return partial_sums[0] / N;
}

void measure_time(double (*f)(const double*, size_t), size_t N, std::unique_ptr<double[]>& arr, string msg) {
	double t1 = omp_get_wtime();
	double v = f(arr.get(), N);
	double t2 = omp_get_wtime();
	std::cout << msg << t2 - t1 << std::endl;
	std::cout << "Result: " << v << std::endl << std::endl;
}

template <class F>
auto measure_time_chrono(F f, size_t N, std::unique_ptr<double[]>& arr) {
	using namespace std::chrono;
	auto t1 = steady_clock::now();
	f(arr.get(), N);
	auto t2 = steady_clock::now();
	return duration_cast<milliseconds>(t2 - t1).count();
}

const uint32_t A = 22695477;
const uint32_t B = 1;

class lc_t {
public:
	uint32_t A, B;

	lc_t(uint32_t a = 1, uint32_t b = 0) : A(a), B(b) {}

	lc_t& operator *= (const lc_t& x) {
		A *= x.A;
		B += A * x.B;
		return *this;
	}

	auto operator() (uint32_t seed, uint32_t min_value, uint32_t max_value) const {
		if (max_value - min_value + 1 != 0) {
			return (A * seed + B) % (max_value - min_value) + min_value;
		}
		else {
			return A * seed + B;
		}
	}
};

double randomize_vector(uint32_t* V, size_t n, size_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	double res = 0.0;

	if (min_val > max_val) {
		exit(__LINE__);
	}

	lc_t generator(A, B);

	for (int i = 1; i < n; ++i) {
		generator *= generator;
		V[i] = generator(seed, min_val, max_val);
		res += V[i];
	}

	return res / n;
}

double randomize_vector_par(uint32_t* V, size_t n, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	double res = 0;

	unsigned T = get_num_threads();
	std::vector<std::thread> workers;
	std::mutex mtx;

	auto worker_proc = [V, n, seed, T, min_val, max_val, &res, &mtx](unsigned t) {
		double partial = 0;
		auto generator = lc_t(A, B);

		size_t b = n % T, e = n / T;
		if (t < b)
			b = t * ++e;
		else
			b += t * e;
		e += b;

		generator = fast_pow(generator, b + 1);;
		for (int i = b; i < e; ++i) {
			generator *= generator;
			V[i] = generator(seed, min_val, max_val);
			partial += V[i];
		}

		{
			std::scoped_lock l{ mtx };
			res += partial;
		}

		};

	for (unsigned t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}

	worker_proc(0);
	for (auto& w : workers) {
		w.join();
	}

	return res / n;
}

double randomize_vector_par(std::vector<uint32_t>& V, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	return randomize_vector_par(V.data(), V.size(), seed, min_val, max_val);
}

int main() {
	const size_t N = 1u << 4; // 2^n

	//auto buf = make_unique<double[]>(N);
	/*for (size_t i = 0; i < N; ++i) {
		buf[i] = i;
	}*/

	std::vector<uint32_t> buf(N);
	randomize_vector_par(buf, 10000, 0, 100);

	for (size_t i = 0; i < N; i++)
	{
		cout << buf[i] << " ";
	}
	cout << "\n";

	std::vector<measure_func> functions_for_measure{
		//measure_func("average", average),
		measure_func("average_reduce", average_reduce),
		//measure_func("average_rr", average_rr),
		//measure_func("average_omp", average_omp),
		measure_func("average_omp_aligned", average_omp_aligned),
		measure_func("average_cpp_aligned", average_cpp_aligned),
		//measure_func("average_omp_mtx", average_omp_mtx),
		//measure_func("average_cpp_mtx", average_cpp_mtx),
		measure_func("average_cpp_mtx_local", average_cpp_mtx_local),
		measure_func("average_cpp_reduction", average_cpp_reduction)
	};

	if (_isatty(_fileno(stdout))) {
		// Код ниже для вывода в консоль
		for (auto& mf : functions_for_measure) {
			auto exp_res = run_experiement_cpp(mf.func, N, buf.data());
			std::cout << "Function: " << mf.name << '\n';
			std::cout << "T\tResult\t\t\tTime\t\tSpeedup\t\t\tEfficiency" << '\n';
			for (auto& ev : exp_res) {
				std::cout << ev.T << "\t";
				std::cout << ev.result << "\t\t";
				std::cout << ev.time << "\t\t";
				std::cout << ev.speedup << "\t\t\t";
				std::cout << ev.efficiency << '\n';
			}
		}
	}
	else {
		cout.imbue(std::locale(""));
		// Код ниже для перенаправленного вывода
		std::cout << "Method;T;Result;Time;Speedup;Efficiency\n";
		for (auto& mf : functions_for_measure) {
			auto exp_res = run_experiement_cpp(mf.func, N, buf.data());
			// Код ниже для вывода в файл
			for (auto& ev : exp_res) {
				std::cout << mf.name << ";";
				std::cout << ev.T << ";";
				std::cout << ev.result << ";";
				std::cout << ev.time << ";";
				std::cout << ev.speedup << ";";
				std::cout << ev.efficiency << "\n";
			}
		}
	}

	return 0;
}