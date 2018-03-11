#pragma once
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

__device__ __forceinline__ float3 mod289(float3 x) { return x - floor(x * (1.f / 289.f)) * 289.f; }
__device__ __forceinline__ float2 mod289(float2 x) { return x - floor(x * (1.f / 289.f)) * 289.f; }
// Permutation polynomial: (34x^2 + x) mod 289
__device__ __forceinline__ float3 permute(float3 x) { return mod289(((x*34.f) + 1.f)*x); }
__device__ __forceinline__ float  fract(float x) { return x - floorf(x); }
__device__ __forceinline__ float2 fract(float2 x) { return make_float2(fract(x.x), fract(x.y)); }
__device__ __forceinline__ float3 fract(float3 x) { return make_float3(fract(x.x), fract(x.y), fract(x.z)); }
__device__ __forceinline__ float3 abs3(float3 x) { return make_float3(::abs(x.x), ::abs(x.y), ::abs(x.z)); }

__device__ __forceinline__ float3 mod7(float3 x) { return x - floor(x * (1.f / 7.f)) * 7.f; }

// Returns scalar simplex noise [-1, 1]
__device__ float snoise(float2 v)
{
	const float4 C = make_float4(
		0.211324865405187f,  // (3.0-sqrt(3.0))/6.0
		0.366025403784439f,  // 0.5*(sqrt(3.0)-1.0)
		-0.577350269189626f,  // -1.0 + 2.0 * C.x
		0.024390243902439f); // 1.0 / 41.0

							 // First corner
	float2 i = floor(v + make_float2(v.x * C.y + v.y * C.y));  // floor(v + dot(v, C.yy) );
	const float2 x0 = v - i + make_float2(i.x * C.x + i.y * C.x);  // v - i + dot(i, C.xx);

																   // Other corners
																   //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
																   //i1.y = 1.0 - i1.x;
	const float2 i1 = (x0.x > x0.y) ? make_float2(1.f, 0.f) : make_float2(0.f, 1.f);
	// x0 = x0 - 0.0 + 0.0 * C.xx ;
	// x1 = x0 - i1 + 1.0 * C.xx ;
	// x2 = x0 - 1.0 + 2.0 * C.xx ;
	const float4 x12 = make_float4(
		x0.x + C.x - i1.x,
		x0.y + C.x - i1.y,
		x0.x + C.z,
		x0.y + C.z); //x0.xyxy + C.xxzz;
					 //x12.xy -= i1;

					 // Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	const float3 p = permute(permute(
		i.y + make_float3(0.f, i1.y, 1.f))
		+ i.x + make_float3(0.f, i1.x, 1.f));

	float3 m = fmaxf(0.5f -
		make_float3(
			x0.x *  x0.x + x0.y *  x0.y,   //dot(x0, x0), 
			x12.x * x12.x + x12.y * x12.y,   //dot(x12.xy, x12.xy), 
			x12.z * x12.z + x12.w * x12.w), //dot(x12.zw, x12.zw)), 
		make_float3(0.f));
	m = m * m;
	m = m * m;

	// Gradients: 41 points uniformly over a line, mapped onto a diamond.
	// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

	const float3 x = 2.f * fract(p * make_float3(C.w)) - 1.f;
	const float3 h = abs3(x) - 0.5f;
	const float3 ox = floor(x + 0.5f);
	const float3 a0 = x - ox;

	// Normalise gradients implicitly by scaling m
	// Approximation of: m *= inversesqrt( a0*a0 + h*h );
	m *= 1.79284291400159f - 0.85373472095314f * (a0*a0 + h * h);

	// Compute final noise value at P
	const float3 g = make_float3(
		a0.x *  x0.x + h.x *  x0.y,
		a0.y * x12.x + h.y * x12.y, //g.yz = a0.yz * x12.xz + h.yz * x12.yw;
		a0.z * x12.z + h.z * x12.w);

	return 130.f * dot(m, g);
}



// Cellular noise ("Worley noise") in 2D in GLSL.
// Copyright (c) Stefan Gustavson 2011-04-19. All rights reserved.
// This code is released under the conditions of the MIT license.
// See LICENSE file for details.

// Cellular noise, returning F1 and F2 in a float2.
// Standard 3x3 search window for good F1 and F2 values
__device__ float2 cellular(float2 P)
{
	const float K = 0.142857142857f; // 1/7
	const float Ko = 0.428571428571f; // 3/7
	const float jitter = 1.f; // Less gives more regular pattern
	float2 Pi = mod289(floor(P));
	float2 Pf = fract(P);
	float3 oi = make_float3(-1.f, 0.f, 1.f);
	float3 of = make_float3(-0.5f, 0.5f, 1.5f);
	float3 px = permute(Pi.x + oi);
	float3 p = permute(px.x + Pi.y + oi); // p11, p12, p13
	float3 ox = fract(p*K) - Ko;
	float3 oy = mod7(floor(p*K))*K - Ko;
	float3 dx = Pf.x + 0.5f + jitter * ox;
	float3 dy = Pf.y - of + jitter * oy;
	float3 d1 = dx * dx + dy * dy; // d11, d12 and d13, squared
	p = permute(px.y + Pi.y + oi); // p21, p22, p23
	ox = fract(p*K) - Ko;
	oy = mod7(floor(p*K))*K - Ko;
	dx = Pf.x - 0.5f + jitter * ox;
	dy = Pf.y - of + jitter * oy;
	float3 d2 = dx * dx + dy * dy; // d21, d22 and d23, squared
	p = permute(px.z + Pi.y + oi); // p31, p32, p33
	ox = fract(p*K) - Ko;
	oy = mod7(floor(p*K))*K - Ko;
	dx = Pf.x - 1.5f + jitter * ox;
	dy = Pf.y - of + jitter * oy;
	float3 d3 = dx * dx + dy * dy; // d31, d32 and d33, squared

								   // Sort out the two smallest distances (F1, F2)
	float3 d1a = fminf(d1, d2);
	d2 = fmaxf(d1, d2); // Swap to keep candidates for F2
	d2 = fminf(d2, d3); // neither F1 nor F2 are now in d3
	d1 = fminf(d1a, d2); // F1 is now in d1
	d2 = fmaxf(d1a, d2); // Swap to keep candidates for F2
	if (d1.x >= d1.y) { float t = d1.x; d1.x = d1.y; d1.y = t; } //d1.xy = (d1.x < d1.y) ? d1.xy : d1.yx; // Swap if smaller
	if (d1.x >= d1.z) { float t = d1.x; d1.x = d1.z; d1.z = t; } //d1.xz = (d1.x < d1.z) ? d1.xz : d1.zx; // F1 is in d1.x
	d1.y = fminf(d1.y, d2.y); d1.z = fminf(d1.z, d2.z); //d1.yz = fminf(d1.yz, d2.yz); // F2 is now not in d2.yz
	d1.y = fminf(d1.y, d1.z); // nor in d1.z
	d1.y = fminf(d1.y, d2.x); // F2 is in d1.y, we're done.
	return make_float2(sqrtf(d1.x), sqrtf(d1.y)); //sqrt(d1.xy);
}

__device__ __forceinline__ float cellular1(float2 P)
{
	const float2 ff = cellular(P);
	return ff.y - ff.x;
}
