#pragma once

#include <optix.h>

rtBuffer<int, 1> primes;

__device__ __forceinline__ int rev(const int i, const int p) { if (i == 0) return i; else return p - i; }

__device__ __forceinline__ float hal(const int b, uint& j) { // Halton sequence with reverse permutation
	const int p = primes[b]; float h = 0.0f, f = 1.0f / (float)p, fct = f;
	while (j > 0) { h += rev(j % p, p) * fct; j /= p; fct *= f; } return h;
}