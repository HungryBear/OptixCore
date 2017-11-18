
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

#include <optixu/optixu_vector_types.h>

typedef struct struct_BasicLight
{
#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif
  float3 pos;
  float3 color;
  int    casts_shadow; 
  int    padding;      // make this structure 32 bytes -- powers of two are your friend!
} BasicLight;

struct TriangleLight
{
#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif
  float3 v1, v2, v3;
  float3 normal;
  float3 emission;
  int idx;
};

struct Lightsource
{
	int type;
	int idx;
	float3 emission;

	
	struct trianglelight
	{
		float3 v0, v1, v2;
		float3 normal;
	};
	struct pointlight
	{
		float3 pos;
		float  radius;
	};
	struct dirlight
	{
		float3 dir;
		float  solid_angle;
	};
	struct envmap
	{
		int map_id;
	};
	union {
		trianglelight triLight;
		envmap envMap;
		pointlight pointLight;
	};

};

struct PerRayData_pathtrace_shadow
{
	bool inShadow;
	float3 attenuation;
	bool inside;
	float dist;
	int lightIndex;
	unsigned int seed;
};
