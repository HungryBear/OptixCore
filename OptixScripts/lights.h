#pragma once

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_vector_types.h>

#include "scene.h"
#include "helpers.h"
#include "math_utils.h"
#include "lightsampling.h"

#define PointLight 1
#define AreaLight 2
#define EnvMap 3


using namespace optix;

struct Lightinfo
{
	int idx;
	int type;
	float3 emission;

	int startTriangle;
	int endTriangle;
	int samplerId;
	float3 vData;
	float  fData;
};

struct LightSample
{
	float3 directionToLight;
	float3 radiance;
	float  areaPdf;
	float  emissionPdf;
	float  cosAtLight;
	int  isDelta, isFinite;
};

struct TriangleInfo
{
	float3 v0, v1, v2, n;
	float  area;
};

rtBuffer<Lightinfo>           lights;
rtBuffer<TriangleInfo>        lightTriangles;
rtDeclareVariable(uint, shadow_ray_type, , );


__device__ __forceinline__ int SampleIndex(const float& u0, const int max)
{
	if (max <= 1)		return 0;
	return int(floorf(max*u0));
}

__device__ __forceinline__ int SampleIndex(const float& u0, const int imin, const int imax)
{
	return min(imax - 1, imin + SampleIndex(u0, imax - imin));
}

__device__ __inline__ float3 trilight_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li,
	const float& u0, const float& u1, const float& u2,
	optix::Ray& shadow_ray, float& aPdf, float& ePdf, float& cosAt)
{
	float pn = 1.0f / (float)(li.endTriangle - li.startTriangle);
	const TriangleInfo tr = lightTriangles[SampleIndex(u0, li.startTriangle, li.endTriangle)];

	aPdf = pn;
	float2       aRndTuple = make_float2(u1, u2);
	float3             oDirectionToLight;
	float             oDistance;

	float3 r = IlluminateAreaLight(tr.v0, tr.v1, tr.v2, tr.n, li.emission, pos, aRndTuple, oDirectionToLight, oDistance, aPdf, &ePdf, &cosAt);
	shadow_ray = make_Ray(pos, oDirectionToLight, shadow_ray_type, scene_epsilon, oDistance);
	return r;

	/*
	const float2 uv = SampleUniformTriangle(make_float2(u1, u2));
	const float3 e1 = tr.v1 - tr.v2;
	const float3 e2 = tr.v0 - tr.v2;

	const float3 lightPoint = tr.v0 + e1 * uv.x + e2 * uv.y;
	const float3 l = lightPoint - pos;

	const float distSqr = dot(l, l);
	float  oDistance = sqrtf(distSqr);
	float3 oDirectionToLight = oDirectionToLight / oDistance;

	const float cosNormalDir = dot(n, -oDirectionToLight);
	const float mInvArea = pa;// 1.f / length(cross(e1, e2));

	if (cosAt < EPS_COSINE) 		{ aPdf = 0.0f; 			return make_float3(0.0f); }
	float3 result = li.emission;

	aPdf = pn*mInvArea * distSqr / cosNormalDir;
	cosAt = cosNormalDir;
	ePdf = mInvArea * cosNormalDir * INV_PI_F;

	shadow_ray = make_Ray(pos, oDirectionToLight, pathtrace_shadow_ray_type, scene_epsilon, oDistance);
	return result;
	*/
}

__device__ __inline__ float3 envmap_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li,
	const float& u0, const float& u1,
	optix::Ray& shadow_ray, float& aPdf, float& ePdf, float& cosAt)
{
	cosAt = 1.0f;
	float oDirectPdfW = 1.0f;
	float3 oDirectionToLight = SampleUniformSphereW(make_float2(u0, u1), &oDirectPdfW);
	// This stays even with image sampling
	float oDistance = scene_sphere_radius * 2;
	float oEmissionPdfW = oDirectPdfW * ConcentricDiscPdfA() * 1.0f / (scene_sphere_radius*scene_sphere_radius);
	float2 t = ll_map(normalize(oDirectionToLight));
	float3 lAttenuation = make_float3(rtTex2D<float4>(li.samplerId, t.x, t.y))*abs(dot(n, oDirectionToLight));
	shadow_ray = make_Ray(pos, normalize(oDirectionToLight), shadow_ray_type, scene_epsilon, oDistance);
	aPdf = oDirectPdfW;
	ePdf = oEmissionPdfW;
	return lAttenuation;
}

__device__ __inline__ float3 pointlight_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li,
	optix::Ray& shadow_ray, float& aPdf, float& ePdf, float& cosAt)
{
	float3 L = (li.vData - pos);
	float nmDl = dot(n, normalize(L));
	cosAt = 1.0f;
	ePdf = 1.0f;
	if (nmDl > 0.0f)
	{
		float Ldist = length(L);
		shadow_ray = make_Ray(pos, normalize(L), shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
		aPdf = nmDl / Ldist * Ldist;
		return li.emission;
	}
	else {
		aPdf = 0.0f;
	}
	return make_float3(0);
}

__device__ __inline__ float3 SampleSurfaceShadowRay(const float4& sample, const float3& pos, const float3& n, Ray& shadow_ray, LightSample* ls, float* lightPickPdf)
{
	const Lightinfo li = lights[max(0, SampleIndex(sample.x, lights.size()))];
	*lightPickPdf = 1.0f / float(lights.size());
	if (li.type == AreaLight)
	{
		ls->isDelta = 0;
		ls->isFinite = 1;
		return trilight_sample_shadowray(pos, n, li, sample.x, sample.y, sample.z, shadow_ray, ls->areaPdf, ls->emissionPdf, ls->cosAtLight);
	}
	else
		if (li.type == EnvMap)
		{
			ls->isDelta = 0;
			ls->isFinite = 0;
			return envmap_sample_shadowray(pos, n, li, sample.x, sample.y, shadow_ray, ls->areaPdf, ls->emissionPdf, ls->cosAtLight);
		}
		else
			if (li.type == PointLight)
			{
				ls->isDelta = 1;
				ls->isFinite = 1;
				return pointlight_sample_shadowray(pos, n, li, shadow_ray, ls->areaPdf, ls->emissionPdf, ls->cosAtLight);
			}
	return make_float3(0.0f);

}


__device__ __inline__ float3 Scene_Emit(const float u0, const float4& sample, float3& pos, float3& dir, LightSample* ls)
{
	const Lightinfo li = lights[SampleIndex(u0, lights.size())];
	float lightPickPdf = 1.0f / float(lights.size());
	float3 result = make_float3(0);
	ls->cosAtLight = 1.0f;
	if (li.type == AreaLight)
	{
		ls->isDelta = 0;
		ls->isFinite = 1;
		float pn = 1.0f / (float)(li.endTriangle - li.startTriangle);
		const TriangleInfo tr = lightTriangles[SampleIndex(sample.x, li.startTriangle, li.endTriangle)];

		float		aPdf = 0;
		float2       aRndTuple = make_float2(sample.y, sample.z);
		float2       aPosRndTuple = make_float2(sample.w, u0);

		result = EmitAreaLight(tr.v0, tr.v1, tr.v2, tr.n, li.emission, aRndTuple, aPosRndTuple, pos, dir, ls->emissionPdf, &aPdf, &ls->cosAtLight);
		aPdf *= pn * lightPickPdf;
		ls->emissionPdf *= pn * lightPickPdf;
		ls->areaPdf = aPdf;

	}
	else
		if (li.type == EnvMap)
		{
			ls->isDelta = 0;
			ls->isFinite = 0;

			float directPdf;
			float2       aDirRndTuple = make_float2(sample.x, sample.y);
			float2       aPosRndTuple = make_float2(sample.z, sample.w);

			// Replace these two lines with image sampling
			float3 oDirection = SampleUniformSphereW(aDirRndTuple, &directPdf);
			const float2 xy = SampleConcentricDisc(aPosRndTuple);
			Frame frame = Frame(oDirection);

			pos = scene_sphere_center + scene_sphere_radius * (-oDirection + frame.mX*xy.x + frame.mY*xy.y);
			dir = oDirection;
			float2 t = ll_map(normalize(oDirection));
			result = make_float3(rtTex2D<float4>(li.samplerId, t.x, t.y));

			ls->emissionPdf = directPdf * lightPickPdf*ConcentricDiscPdfA()*(1.0f / (scene_sphere_radius*scene_sphere_radius));
			ls->areaPdf = lightPickPdf * directPdf;


		}
		else if (li.type == PointLight)
		{
			ls->isDelta = 1;
			ls->isFinite = 1;

			pos = li.vData;
			float directPdf;
			float2       aDirRndTuple = make_float2(sample.x, sample.y);
			dir = SampleUniformSphereW(aDirRndTuple, &directPdf);
			ls->areaPdf = lightPickPdf;
			ls->emissionPdf = directPdf * lightPickPdf;
			result = li.emission;
		}
	return result;
}
