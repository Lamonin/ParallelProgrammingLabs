#pragma once
#include <thread>
#include <mutex>
#include <omp.h>

unsigned get_num_threads();
void set_num_threads(unsigned T);

class latch {
private:
	unsigned T;
	std::mutex mtx;
	std::condition_variable cv;

public:
	latch(unsigned threads);
	void arrive_and_wait();
};

class barrier {
private:
	unsigned lock_id = 0;
	unsigned T, Tmax;
	std::mutex mtx;
	std::condition_variable cv;

public:
	barrier(unsigned threads);
	void arrive_and_wait();
};
