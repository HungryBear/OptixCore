#pragma once
#include <optix.h>
#include <optix_math.h>

using namespace optix;


__device__ __inline__ unsigned int TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
	unsigned b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
};

__device__ __inline__ unsigned int LCGStep(unsigned &z, unsigned A, unsigned C)
{
	return z = (A*z + C);
};

__device__ __inline__ uint intrnd(uint& seed) // 1<=seed<=m
{
//#if LONG_MAX > (16807*2147483647)
//	uint const a = 16807;      //ie 7**5
//	uint const m = 2147483647; //ie 2**31-1
//	seed = (long(seed * a)) % m;
//	return seed;
//#else
	double const a = 16807;      //ie 7**5
	double const m = 2147483647; //ie 2**31-1

	double temp = seed * a;
	seed = (int)(temp - m * floor(temp / m));
	return seed;
//#endif
}

__device__ __inline__ uint4 create_seed(uint seed)
{
	const uint Y = 842502087u, Z = 3579807591u, W = 273326509u;

	return make_uint4(seed, Y, Z, W);
}

__device__ __inline__ float mrs_rnd(uint4& seed)
{
	uint t = (seed.x ^ (seed.x << 11));
	seed.x = seed.y;
	seed.y = seed.z;
	seed.z = seed.w;

	float c = ((seed.w = (seed.w ^ (seed.w >> 19u)) ^ (t ^ (t >> 8u))) * (1.0f / 4294967295.0f));

	return c;
}

__device__  __inline__ unsigned int hash(unsigned int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}



__device__ __inline__ float randfloat(unsigned& i, unsigned p) {
	i ^= p;
	i ^= i >> 17;
	i ^= i >> 10; i *= 0xb36534e5;
	i ^= i >> 12;
	i ^= i >> 21; i *= 0x93fc4795;
	i ^= 0xdf6e307f;
	i ^= i >> 17; i *= 1 | p >> 18;
	return i * (1.0f / 4294967808.0f);
}
