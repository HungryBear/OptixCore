#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>

#include "math_utils.h"

rtDeclareVariable(float3, mPosition, , );
rtDeclareVariable(float3, mForward, , );
rtDeclareVariable(float2, mResolution, , );
rtDeclareVariable(Matrix4x4, mRasterToWorld, , );
rtDeclareVariable(Matrix4x4, mWorldToRaster, , );
rtDeclareVariable(float, mImagePlaneDist, , );


__host__ __device__ __forceinline__ int RasterToIndex(const float2 &aPixelCoords) 
{
	return int(floorf(aPixelCoords.x) + floorf(aPixelCoords.y) * mResolution.x);
}

__host__ __device__ __forceinline__ float2 IndexToRaster(const int &aPixelIndex) 
{
	const float y = floorf(aPixelIndex / mResolution.x);
	const float x = float(aPixelIndex) - y * mResolution.x;
	return make_float2(x, y);
}

__host__ __device__ __forceinline__ float3 RasterToWorld(const float2 &aRasterXY) 
{
	const float4 result = mRasterToWorld * make_float4(aRasterXY.x, aRasterXY.y, 0, 1);
	return make_float3(result.x / result.w, result.y / result.w, result.z / result.w);
}

__host__ __device__ __forceinline__ float2 WorldToRaster(const float3 &aWorldPos) 
{
	const float4 temp = mWorldToRaster * make_float4(aWorldPos, 1);
	return make_float2(temp.x / temp.w, temp.y / temp.w);
}

// returns false when raster position is outside screen space
__host__ __device__ __forceinline__ bool CheckRaster(const float2 &aRasterPos) 
{
	return aRasterPos.x >= 0 && aRasterPos.y >= 0 &&
		aRasterPos.x < mResolution.x && aRasterPos.y < mResolution.y;
}

__host__ __device__ __forceinline__ void GenerateRay(const float2 &aRasterXY, float3 &org, float3 &dir, float* cameraPdf = NULL) 
{
	const float3 worldRaster = RasterToWorld(aRasterXY);

	org = mPosition;
	dir = normalize(worldRaster - mPosition);
	if (cameraPdf != NULL)
	{

		const float cosAtCamera = dot(mForward, dir);
		const float imagePointToCameraDist = mImagePlaneDist / cosAtCamera;
		const float imageToSolidAngleFactor = (imagePointToCameraDist*imagePointToCameraDist) / cosAtCamera;

		const float cameraPdfW = imageToSolidAngleFactor;
		*cameraPdf = 1.0f / cameraPdfW;
	}
}



