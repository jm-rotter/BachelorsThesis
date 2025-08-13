#include "cpu_matmul.h"
#include "gpu_matmul.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>


int main() {

	int a, b, c;

	a = 16;
	b = 16;
	c = 16;

	auto t1 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < 32; i++) {
		Matrix mat1 = createMatrix(a, b);
		Matrix mat2 = createMatrix(b, c);
		Matrix gpuNaiveRes = createMatrix(a, c);
		Matrix cpuResult = createMatrix(a,c);
		fill_random(mat1);
		fill_random(mat2);
//		printf("Executing task: %d\n", i);
		gpu_matmul(mat1,mat2, gpuNaiveRes, NAIVE);
		cpu_matmul(mat1, mat2, cpuResult);
//		std::cout << "Comparison: " << (compare(gpuNaiveRes, cpuResult) ? "CPU result = GPU Naive result" : "CPU result != GPU Naive Result") << "\n";
		compare(gpuNaiveRes, cpuResult);
	}
	cudaDeviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();
	double gpuNaiveTime = time_in_ms(t1, t2);
	std::cout << "GPU Non Persistent Time: " << gpuNaiveTime << " ms\n";



}
