
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "random.h"

using namespace optix;

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   vindex_buffer;    // position indices 
rtBuffer<int3>   nindex_buffer;    // normal indices
rtBuffer<int3>   tindex_buffer;    // texcoord indices

//rtBuffer<uint>   material_buffer; // per-face material index
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


rtDeclareVariable(float3, v1, attribute v1, );
rtDeclareVariable(float3, v2, attribute v2, );
rtDeclareVariable(float3, v3, attribute v3, );

RT_PROGRAM void mesh_intersect(int primIdx)
{
	int3 v_idx = vindex_buffer[primIdx];

	float3 p0 = vertex_buffer[v_idx.x];
	float3 p1 = vertex_buffer[v_idx.y];
	float3 p2 = vertex_buffer[v_idx.z];


	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma, delta;
	if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {

		if (rtPotentialIntersection(t))
		{
			v1 = p0;
			v2 = p1;
			v3 = p2;

			if (normal_buffer.size() == 0 || nindex_buffer.size() == 0) {
				shading_normal = -n;
			}
			else {
				int3 n_idx = nindex_buffer[primIdx];
				if (n_idx.x < 0 || n_idx.y < 0 || n_idx.z < 0) {
					shading_normal = -n;
				}
				else {
					float3 n0 = normal_buffer[n_idx.x];
					float3 n1 = normal_buffer[n_idx.y];
					float3 n2 = normal_buffer[n_idx.z];
					shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));
				}
			}
			geometric_normal = -n;

			if (texcoord_buffer.size() == 0 || tindex_buffer.size() == 0) {
				texcoord = make_float3(0.0f, 0.0f, 0.0f);
			}
			else {
				int3 t_idx = tindex_buffer[primIdx];
				float2 t0 = texcoord_buffer[t_idx.x];
				float2 t1 = texcoord_buffer[t_idx.y];
				float2 t2 = texcoord_buffer[t_idx.z];
				delta = (1.0f - beta - gamma);
				//texcoord = make_float3(t0.x * beta + t1.x * gamma + t2.x * (1.0f - beta - gamma), t0.y * beta + t1.y * gamma + t2.y * (1.0f - beta - gamma), 0.f);

				//texcoord = make_float3(t1*delta + t2*beta+t0* gamma, 0.f);

				//texcoord = make_float3( t2*beta + t1*gamma + t0*(1.0f-beta-gamma) );
				texcoord = make_float3(t1*beta + t2*gamma + t0*(1.0f - beta - gamma));

			}

			rtReportIntersection(0 /*material_buffer[primIdx]*/);
		}
	}
}

__device__ __inline__ float sample_distance(const float e, const float sig_t)
{
	return log(1.f - e) / sig_t;
}

RT_PROGRAM void volume_mesh_intersect(int primIdx)
{
	int3 v_idx = vindex_buffer[primIdx];

	float3 p0 = vertex_buffer[v_idx.x];
	float3 p1 = vertex_buffer[v_idx.y];
	float3 p2 = vertex_buffer[v_idx.z];

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma, delta;

	if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {
		unsigned int i = (uint)primIdx;
		float z1 = rnd(i);
		float dist = sample_distance( z1, 0.5f);
		t += dist;

		if (rtPotentialIntersection(t))
		{

			if (normal_buffer.size() == 0 || nindex_buffer.size() == 0) {
				shading_normal = -n;
			}
			else {
				int3 n_idx = nindex_buffer[primIdx];
				if (n_idx.x < 0 || n_idx.y < 0 || n_idx.z < 0) {
					shading_normal = -n;
				}
				else {
					float3 n0 = normal_buffer[n_idx.x];
					float3 n1 = normal_buffer[n_idx.y];
					float3 n2 = normal_buffer[n_idx.z];
					shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));
				}
			}
			geometric_normal = -n;

			if (texcoord_buffer.size() == 0 || tindex_buffer.size() == 0) {
				texcoord = make_float3(0.0f, 0.0f, 0.0f);
			}
			else {
				int3 t_idx = tindex_buffer[primIdx];
				float2 t0 = texcoord_buffer[t_idx.x];
				float2 t1 = texcoord_buffer[t_idx.y];
				float2 t2 = texcoord_buffer[t_idx.z];
				delta = (1.0f - beta - gamma);
				//texcoord = make_float3(t0.x * beta + t1.x * gamma + t2.x * (1.0f - beta - gamma), t0.y * beta + t1.y * gamma + t2.y * (1.0f - beta - gamma), 0.f);

				//texcoord = make_float3(t1*delta + t2*beta+t0* gamma, 0.f);

				//texcoord = make_float3( t2*beta + t1*gamma + t0*(1.0f-beta-gamma) );
				texcoord = make_float3(t1*beta + t2*gamma + t0*(1.0f - beta - gamma));

			}

			rtReportIntersection(0 /*material_buffer[primIdx]*/);
		}
	}
}


RT_PROGRAM void mesh_bounds(int primIdx, float result[6])
{
	const int3 v_idx = vindex_buffer[primIdx];

	const float3 v0 = vertex_buffer[v_idx.x];
	const float3 v1 = vertex_buffer[v_idx.y];
	const float3 v2 = vertex_buffer[v_idx.z];
	const float  area = length(cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	}
	else {
		aabb->invalidate();
	}
}

