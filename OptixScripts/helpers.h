/*
* Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#pragma once

#include <optix_math.h>

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __CUDACC__
__device__ __inline__ uchar4 make_color(const float3& c)
{
	return make_uchar4(static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
		static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
		static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
		255u);                                                 /* A */
}

__device__ __inline__ uchar4 make_color(const float4& c)
{
	return make_uchar4(static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
		static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
		static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
		static_cast<unsigned char>(__saturatef(c.w)*255.99f)); /* A */
}

__device__ __inline__ uchar4 bgra_to_rgba(const uchar4& c)
{
	return make_uchar4(c.x, c.y, c.z, c.w);
}

__device__ __inline__ uchar4 rgba_to_bgra(const uchar4& c)
{
	return make_uchar4(c.z, c.y, c.x, c.w);
}
#endif

// Calculate the luminance value of an rgb triple
__device__ __inline__ float luminance(const float3& rgb)
{
	const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
	return  dot(rgb, ntsc_luminance);
}

__device__ __inline__ float rand22(){
	return 1.0f;
}

// Maps concentric squares to concentric circles (Shirley and Chiu)
__host__ __device__ __inline__ float2 square_to_disk(float2 sample)
{
	float phi, r;

	const float a = 2.0f * sample.x - 1.0f;
	const float b = 2.0f * sample.y - 1.0f;

	if (a > -b)
	{
		if (a > b)
		{
			r = a;
			phi = (float)M_PI_4f * (b / a);
		}
		else
		{
			r = b;
			phi = (float)M_PI_4f * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (float)M_PI_4f * (4.0f + (b / a));
		}
		else
		{
			r = -b;
			phi = (b) ? (float)M_PI_4f * (6.0f - (a / b)) : 0.0f;
		}
	}

	return make_float2(r * cosf(phi), r * sinf(phi));
}

// Convert cartesian coordinates to polar coordinates
__host__ __device__ __inline__ float3 cart_to_pol(float3 v)
{
	float azimuth;
	float elevation;
	float radius = length(v);

	float r = sqrtf(v.x*v.x + v.y*v.y);
	if (r > 0.0f)
	{
		azimuth = atanf(v.y / v.x);
		elevation = atanf(v.z / r);

		if (v.x < 0.0f)
			azimuth += M_PIf;
		else if (v.y < 0.0f)
			azimuth += M_PIf * 2.0f;
	}
	else
	{
		azimuth = 0.0f;

		if (v.z > 0.0f)
			elevation = +M_PI_2f;
		else
			elevation = -M_PI_2f;
	}

	return make_float3(azimuth, elevation, radius);
}

// Sample Phong lobe relative to U, V, W frame
__host__ __device__ __inline__ float3 sample_phong_lobe(float2 sample, float exponent, float3 U, float3 V, float3 W)
{
	const float power = expf(logf(sample.y) / (exponent + 1.0f));
	const float phi = sample.x * 2.0f * (float)M_PIf;
	const float scale = sqrtf(1.0f - power * power);

	const float x = cosf(phi)*scale;
	const float y = sinf(phi)*scale;
	const float z = power;

	return x * U + y * V + z * W;
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void create_onb(const float3& n, float3& U, float3& V, float3& W)
{
	W = normalize(n);
	U = cross(W, make_float3(0.0f, 1.0f, 0.0f));

	if (fabs(U.x) < 0.001f && fabs(U.y) < 0.001f && fabs(U.z) < 0.001f)
		U = cross(W, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(W, U);
}

// Create ONB from normalized vector
__device__ __inline__ void create_onb(const float3& n, float3& U, float3& V)
{
	U = cross(n, make_float3(0.0f, 1.0f, 0.0f));

	if (dot(U, U) < 1e-3f)
		U = cross(n, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(n, U);
}

// Compute the origin ray differential for transfer
__host__ __device__ __inline__ float3 differential_transfer_origin(float3 dPdx, float3 dDdx, float t, float3 direction, float3 normal)
{
	float dtdx = -dot((dPdx + t * dDdx), normal) / dot(direction, normal);
	return (dPdx + t * dDdx) + dtdx * direction;
}

// Compute the direction ray differential for a pinhole camera
__host__ __device__ __inline__ float3 differential_generation_direction(float3 d, float3 basis)
{
	float dd = dot(d, d);
	return (dd*basis - dot(d, basis)*d) / (dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
__host__ __device__ __inline__ float3 differential_reflect_direction(float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N)
{
	float3 dNdx = dNdP * dPdx;
	float dDNdx = dot(dDdx, N) + dot(D, dNdx);
	return dDdx - 2 * (dot(D, N)*dNdx + dDNdx * N);
}

// Compute the direction ray differential for refraction
__host__ __device__ __inline__ float3 differential_refract_direction(float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N, float ior, float3 T)
{
	float eta;
	if (dot(D, N) > 0.f) {
		eta = ior;
		N = -N;
	}
	else {
		eta = 1.f / ior;
	}

	float3 dNdx = dNdP * dPdx;
	float mu = eta * dot(D, N) - dot(T, N);
	float TN = -sqrtf(1 - eta * eta*(1 - dot(D, N)*dot(D, N)));
	float dDNdx = dot(dDdx, N) + dot(D, dNdx);
	float dmudx = (eta - (eta*eta*dot(D, N)) / TN)*dDNdx;
	return eta * dDdx - (mu*dNdx + dmudx * N);
}

template <class T>
__host__ __device__ __inline__ float3 bilerp(float u, float v,
	const T& x00, const T& x10,
	const T& x01, const T& x11)
{
	return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

__host__ __device__ __inline__ float3 Yxy2XYZ(const float3& Yxy)
{
	return make_float3(Yxy.y * (Yxy.x / Yxy.z),
		Yxy.x,
		(1.0f - Yxy.y - Yxy.z) * (Yxy.x / Yxy.z));
}

__host__ __device__ __inline__ float3 XYZ2rgb(const float3& xyz)
{
	const float R = dot(xyz, make_float3(3.2410f, -1.5374f, -0.4986f));
	const float G = dot(xyz, make_float3(-0.9692f, 1.8760f, 0.0416f));
	const float B = dot(xyz, make_float3(0.0556f, -0.2040f, 1.0570f));
	return make_float3(R, G, B);
}

__host__ __device__ __inline__ float3 Yxy2rgb(float3 Yxy)
{
	// First convert to xyz
	float3 xyz = make_float3(Yxy.y * (Yxy.x / Yxy.z),
		Yxy.x,
		(1.0f - Yxy.y - Yxy.z) * (Yxy.x / Yxy.z));

	const float R = dot(xyz, make_float3(3.2410f, -1.5374f, -0.4986f));
	const float G = dot(xyz, make_float3(-0.9692f, 1.8760f, 0.0416f));
	const float B = dot(xyz, make_float3(0.0556f, -0.2040f, 1.0570f));
	return make_float3(R, G, B);
}


__host__ __device__ __inline__ float3 rgb2Yxy(float3 rgb)
{
	// convert to xyz
	const float X = dot(rgb, make_float3(0.4124f, 0.3576f, 0.1805f));
	const float Y = dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
	const float Z = dot(rgb, make_float3(0.0193f, 0.1192f, 0.9505f));

	// convert xyz to Yxy
	return make_float3(Y,
		X / (X + Y + Z),
		Y / (X + Y + Z));
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void createONB(const float3& n, float3& U, float3& V, float3& W)
{
	W = normalize(n);
	U = cross(W, make_float3(0.0f, 1.0f, 0.0f));

	if (abs(U.x) < 0.001f && abs(U.y) < 0.001f && abs(U.z) < 0.001f)
		U = cross(W, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(W, U);
}

// sample hemisphere with cosine density
__device__ __inline__ float3 sampleUnitHemisphere(const float2& sample,
	const float3& U,
	const float3& V,
	const float3& W)
{
	float phi = 2.0f * M_PIf * sample.x;
	float r = sqrt(sample.y);
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrtf(max(0.0f, 1.0f - x * x - y * y));

	return x * U + y * V + z * W;
}

__device__ __inline__ float3 norm_rgb(const float4& c)
{
	return make_float3(c.z, c.y, c.x);
}

__device__ __inline__ float3 pow3f(float3 x, float y)
{
	x.x = powf(x.x, y);
	x.y = powf(x.y, y);
	x.z = powf(x.z, y);

	return x;
}


