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

using namespace std;


double average(const double* V, size_t n) {
	double res = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res += V[i];
	}
	return res / n;
}

double average_reduce(const double* V, size_t n) {
	double res = 0.0;
#pragma omp parallel for reduction(+:res)
	for (int i = 0; i < n; ++i) {
		res += V[i];
	}
	return res / n;
}

double average_rr(const double* V, size_t n) {
	double res = 0.0;

#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_num_threads();
		for (int i = t; i < n; i += T) {
			res += V[i]; // Гонка потоков (на этом кончилась пара)
		}
	}

	return res / n;
}

double average_omp(const double* V, size_t n) {
	// double res = 0.0, * partial_sums = (double*)calloc(omp_get_num_procs(), sizeof(double));
	double res = 0.0, * partial_sums;

#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_num_threads();
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

double average_omp_aligned(const double* V, size_t n) {
	unsigned T;
	double res = 0.0;
	partial_sum_t* partial_sums;

#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_num_threads();
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

double average_cpp_aligned(const double* V, size_t n) {
	unsigned T;
	double res = 0.0;
	std::unique_ptr<partial_sum_t[]> partial_sums;
#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_num_threads();
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

double average_omp_mtx(const double* V, size_t n) {
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

double average_cpp_mtx(const double* V, size_t N) {
	double res = 0.0;
	unsigned T = std::thread::hardware_concurrency();
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
double average_cpp_mtx_local(const double* V, size_t N) {
	double res = 0.0;
	unsigned T = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;
	std::mutex mtx;
	size_t ndt = N / T;
	size_t nmt = N % T;

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

double average_cpp_reduction(const double* V, size_t N) {
	unsigned T = std::thread::hardware_concurrency();

	std::vector<double> partial_sums;
	partial_sums.resize(T);

	std::vector<std::thread> workers;

	size_t ndt = N / T;
	size_t nmt = N % T;

	barrier	bar(T);

	auto worker_proc = [&bar, &partial_sums, T, V, N](unsigned t) {
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


// Монитор
// - События (Events)
// - Переменные условия ( Condition variable - cv)
//

// Ускорение 25.10.2023
// То, насколько увеличивается эффективность программы в зависимости
// от роста числа процессоров относительно масштабности задачи
// Speedup = time_t1 / time_tT
// Efficiency = Speedup / T

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

template<typename ReturnType, typename... Args>
ReturnType CallAnyFunc(ReturnType(*f)(Args...), Args... args) {
	return f(args...);
}

void QueueDemo() {
	unsigned P = std::thread::hardware_concurrency();
	unsigned producers_count = 1, consumers_count = P - 1;

	std::vector<std::thread> producers;
	std::vector<std::thread> consumers;

	std::mutex mtx;
	std::queue<int> q;
	std::condition_variable cv;

	auto producer_proc = [&mtx, &q, &cv, consumers_count]() {
		for (unsigned c = 0; c < consumers_count; ++c) {
			std::scoped_lock l{ mtx };
			q.push(c);
			cv.notify_one();
		}
		};

	auto consumer_proc = [&mtx, &cv, &q](unsigned t) {
		std::unique_lock ul(mtx);

		while (q.empty()) {
			cv.wait(ul);
		}

		int m = q.front();
		q.pop();

		cout << "Thread " << t << " received message " << m << "\n";
		};

	for (unsigned i = 0; i < producers_count; ++i) {
		producers.emplace_back(producer_proc);
	}

	for (unsigned i = 0; i < consumers_count; ++i) {
		consumers.emplace_back(consumer_proc, producers_count + i + 1);
	}

	for (auto& p : producers) {
		p.join();
	}

	for (auto& c : consumers) {
		c.join();
	}
}


// 22.11.23
// Барьер памяти (memory fence) (mfence)
// Объект синхронизации 
// C++20:
// std::latch - одноразовый барьер
// latch(T);
// arrive_and_wait();
// std::barrier - переиспользуемый
// barrier(T);
// arrive_and_wait();

struct measure_func {
	std::string name;
	double (*func)(const double*, size_t);
	measure_func(std::string name, double (*func)(const double*, size_t)) : name(name), func(func)
	{
	}
};

int main() {
	size_t N = 1u << 30; // 2^n

	auto buf = make_unique<double[]>(N);
	for (size_t i = 0; i < N; ++i) {
		buf[i] = i;
	}

	//measure_time(average_cpp_mtx_local, N, buf, "Time taken local: ");
	//measure_time(average_cpp_reduction, N, buf, "Time taken cpp reduction: ");

	////measure_time(average, N, buf, "Time taken common: ");
	////measure_time(average_reduce, N, buf, "Time taken Reduce: ");
	////measure_time(average_rr, N, buf, "Time taken RoundRobin: ");
	////measure_time(average_omp, N, buf, "Time taken OMP: ");
	////measure_time(average_omp_aligned, N, buf, "Time taken OMP-aligned: ");
	////measure_time(average_cpp_aligned, N, buf, "Time taken CPP-aligned: ");
	////measure_time(average_omp_mtx, N, buf, "Time taken OMP-Mutex: ");
	////measure_time(average_cpp_mtx, N, buf, "Time taken CPP-Mutex: ");
	////measure_time(average_cpp_mtx_local, N, buf, "Time taken CPP-Mutex localized: ");

	//std::cout << CallAnyFunc(average_reduce, const_cast<const double*>(buf.get()), N) << std::endl;
	//// std::cout << CallAnyFunc(add, 10, 15) << std::endl;

	//// TODO: сделать вывод в CSV файл

	std::vector<measure_func> functions_for_measure{
		measure_func("average_cpp_reduction", average_cpp_reduction),
		measure_func("average", average),
		measure_func("average_reduce", average_reduce),
		//measure_func("average_rr", average_rr),
		//measure_func("average_omp", average_omp),
		//measure_func("average_omp_aligned", average_omp_aligned),
		//measure_func("average_cpp_aligned", average_cpp_aligned),
		//measure_func("average_omp_mtx", average_omp_mtx),
		//measure_func("average_cpp_mtx", average_cpp_mtx),
		//measure_func("average_cpp_mtx", average_cpp_mtx),
		measure_func("average_cpp_mtx_local", average_cpp_mtx_local)
	};

	for (auto& mf : functions_for_measure) {
		auto exp_res = run_experiement_cpp(mf.func, N, buf);

		if (_isatty(_fileno(stdout))) {
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
		else {
			std::cout << "Function:;" << mf.name << '\n';
			std::cout << "T;Result;Time;Speedup;Efficiency\n";
			for (auto& ev : exp_res) {
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