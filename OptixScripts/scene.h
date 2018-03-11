#pragma once
//#define QMC

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"
#include "halton.h"
#include "random.h"
#include "randlib.h"

using namespace optix;

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float, scene_sphere_radius, , );
rtDeclareVariable(float3, scene_sphere_center, , );

//geometry attributes
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );


rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );

rtDeclareVariable(float3, v1, attribute v1, );
rtDeclareVariable(float3, v2, attribute v2, );
rtDeclareVariable(float3, v3, attribute v3, );

rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


__device__ __forceinline__ float Random(uint& seed, const int in = 1, const int prd_depth = 0)
{
	/*
	thrust::default_random_engine rng(seed);
	rng.discard(seed);
	thrust::uniform_real_distribution<float> dst(0, 1);

	return dst(rng);
	*/


#ifdef QMC

	const int sz = primes.size() - 1;
	const int idx = in * max(prd_depth, 1);
	uint sd = seed;
	float sample = idx < sz ? fold(hal(idx, sd)) : fold(hal(idx%sz, sd));
	return sample;
#else
	return randfloat(seed, in*max(prd_depth, 1));
#endif

	//return fold(Sobol2(in*max(prd_depth, 1), seed));
	//return (VanDerCorput(in, seed));
	//return RadicalInverse(seed, 2);
	//return fold(randfloat(seed, in));
};