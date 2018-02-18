#pragma once

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_vector_types.h>

#include "scene.h"
#include "helpers.h"
#include "math_utils.h"

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
	float  lPdf;
	float  ePdf;
	float  geomTerm;
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
rtDeclareVariable(unsigned int, envmapId, , );

__device__ __inline__ float2 ll_map(const float3& dir)
{
	float3 direction = normalize(dir);
	float theta = atan2f(direction.x, direction.y);
	float phi = M_PIf * 0.5f - acosf(direction.z);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sinf(phi));
	return make_float2(u, v);
}


__device__ __inline__ int sampleIndex(const float u0, const int imin, const int imax)
{
	return imin + min(imax - imin, max(0, (int)roundf(u0*(imax - imin))));
}

__device__ float3 trilight_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li, const float& u0, const float& u1, const float& u2,
	optix::Ray& shadow_ray, float& aPdf, float& aG, float& cosAt)
{
	float pn = 1.0f / (float)(li.endTriangle - li.startTriangle);
	const TriangleInfo tr = lightTriangles[sampleIndex(u0, li.startTriangle, li.endTriangle)];
	float pa = 1.0f / tr.area;


	const float2 uv = SampleUniformTriangle(make_float2(u1, u2));
	const float3 e1 = tr.v1 - tr.v2;
	const float3 e2 = tr.v0 - tr.v2;

	const float3 lightPoint = tr.v0 + e1 * uv.x + e2 * uv.y;
	const float3 l = lightPoint - pos;

	const float G = geomFactor(pos, n, lightPoint, tr.n);
	cosAt = dot(n, -l);
	aG = G;
	if (cosAt < 0.0f) { aPdf = 0.0f; 			return make_float3(0.0f); }
	float3 result = li.emission;
	const float sqDist = length(l)*length(l);

	aPdf *= pn * pa;
	//aPdf *= pn*(pa*sqDist / cosAt);

	shadow_ray = make_Ray(pos, normalize(l), shadow_ray_type, scene_epsilon, length(l) - 0.1f);
	return result;
}

__device__ float3 envmap_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li, const float& u0, const float& u1, optix::Ray& shadow_ray, float& aPdf)
{
	float oDirectPdfW = 1.0f;
	float3 oDirectionToLight = SampleUniformSphereW(make_float2(u0, u1), &oDirectPdfW);
	// This stays even with image sampling
	float oDistance = scene_sphere_radius;
	float oEmissionPdfW = oDirectPdfW * ConcentricDiscPdfA() * 1.0f / (scene_sphere_radius*scene_sphere_radius);
	float2 t = ll_map(normalize(oDirectionToLight));
	float3 lAttenuation = make_float3(rtTex2D<float4>(envmapId, t.x, t.y))*dot(n, oDirectionToLight);
	shadow_ray = make_Ray(pos, normalize(oDirectionToLight), shadow_ray_type, scene_epsilon, oDistance);
	//aPdf *= oEmissionPdfW;
	return lAttenuation;
}

__device__ float3 pointlight_sample_shadowray(const float3& pos, const float3& n, const Lightinfo& li, optix::Ray& shadow_ray, float& aPdf)
{
	float3 L = (li.vData - pos);
	float nmDl = dot(n, normalize(L));
	if (nmDl > 0.0f)
	{
		float Ldist = length(L);
		shadow_ray = make_Ray(pos, normalize(L), shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);

		float3 lAttenuation = li.emission;
		aPdf *= nmDl / Ldist * Ldist;
		return (lAttenuation);
	}
	else {
		aPdf = 0.0f;
	}
	return make_float3(0);
}

__device__ float3 SampleSurfaceShadowRay(const float4& sample, const float3& pos, const float3& n, Ray& shadow_ray, float& aPdf, float& aG)
{
	const Lightinfo li = lights[sampleIndex(sample.x, 0, lights.size() - 1)];
	aPdf = 1.0f / lights.size();
	aG = 1.0f;
	float cost;
	if (li.type == AreaLight)
	{
		return trilight_sample_shadowray(pos, n, li, sample.x, sample.y, sample.z, shadow_ray, aPdf, aG, cost);
	}
	else
		if (li.type == EnvMap)
		{
			return envmap_sample_shadowray(pos, n, li, sample.x, sample.y, shadow_ray, aPdf);
		}
		else
			if (li.type == PointLight)
			{
				return pointlight_sample_shadowray(pos, n, li, shadow_ray, aPdf);
			}
			else {
				rtPrintf("Invalid light type");
			}
			return make_float3(0.0f);
}

__device__ __inline__ float3 Emit(const float2 uS, const float4 sample, float3& pos, float3& dir, float& pA, float& pE, float& cosAtLight, Lightinfo& li)
{
	float3 result = make_float3(0.0f);

	li = lights[sampleIndex(uS.x, 0, lights.size() - 1)];
	float aPdf = 1.0f / lights.size();
	cosAtLight = 1.0f;
	pE = 1.0f;
	if (li.type == AreaLight)
	{
		const float pn = 1.0f / (float)(li.endTriangle - li.startTriangle);
		const TriangleInfo tr = lightTriangles[sampleIndex(uS.y, li.startTriangle, li.endTriangle)];
		const float2 uv = SampleUniformTriangle(make_float2(sample.x, sample.y));
		const float invArea = 1.0f / tr.area;

		const float3 e1 = tr.v1 - tr.v2;
		const float3 e2 = tr.v0 - tr.v2;

		pE *= pn * invArea*aPdf;

		aPdf *= pn * invArea;

		const float3 lightPoint = tr.v0 + e1 * uv.x + e2 * uv.y;
		pos = lightPoint;
		float3 d, v1, v2;
		cosine_sample_hemisphere(sample.z, sample.w, d);
		d.z = max(d.z, EPS_COSINE);
		cosAtLight = d.z;
		create_onb(tr.n, v1, v2);
		dir = v1 * d.x + v2 * d.y + tr.n * d.z;
		pA = aPdf;
		result += li.emission*d.z;
	}
	else
		if (li.type == PointLight) {
			pos = li.vData;
			pA = aPdf;
			float3 d = sampleUnitSphere(make_float2(sample.x, sample.y));
			dir = d;
			pE = aPdf;
			result += li.emission;
		}
		else if (li.type == EnvMap) {
			float3 worldCenter = scene_sphere_center;
			float worldRadius = scene_sphere_radius * 1.01f;

			float directPdf;

			dir = SampleUniformSphereW(make_float2(uS.y, sample.x), &directPdf);

			Frame frame(dir);
			float2 xy = SampleConcentricDisc(make_float2(sample.y, sample.z));

			pos = worldCenter + worldRadius * (-dir + frame.mX*xy.x + frame.mY*xy.y);

			pE = directPdf * aPdf*ConcentricDiscPdfA()*(1.0f / worldRadius);
			pA = aPdf;

			float2 t = ll_map(normalize(dir));
			float3 lAttenuation = make_float3(rtTex2D<float4>(1, t.x, t.y));
			pE *= luminanceCIE(lAttenuation);
			result += lAttenuation;
		}

		//else

		return result;
}

__device__ __inline__ float3 sampleLights(const float4 sample, float3& pos, float3& dir, float& pA, float& pE)
{
	float3 result = make_float3(0.0f);

	const Lightinfo li = lights[sampleIndex(sample.x, 0, lights.size() - 1)];
	float aPdf = 1.0f / lights.size();


	if (li.type == AreaLight)
	{
		float pn = 1.0f / (float)(li.endTriangle - li.startTriangle);
		const TriangleInfo tr = lightTriangles[sampleIndex(sample.y, li.startTriangle, li.endTriangle)];
		const float2 uv = SampleUniformTriangle(make_float2(sample.z, sample.w));
		const float3 e1 = tr.v1 - tr.v2;
		const float3 e2 = tr.v0 - tr.v2;
		aPdf *= pn;
		const float3 lightPoint = tr.v0 + e1 * uv.x + e2 * uv.y;
		pos = lightPoint;
		float3 d, v1, v2;
		cosine_sample_hemisphere(sample.w, 1.0f - sample.y, d);
		create_onb(tr.n, v1, v2);
		dir = v1 * d.x + v2 * d.y + tr.n * d.z;

		float RdotN = dot(dir, tr.n);
		if (RdotN < 0.f)
		{
			dir *= -1.f;
			RdotN = -RdotN;
		}


		float A = length(cross(tr.v1 - tr.v2, tr.v0 - tr.v2));
		pA = (RdotN / M_PIf)*(1.0f / A);
		pE = aPdf;
		result += (li.emission / ((1.0f / A)*(RdotN / M_PIf)))*aPdf;
	}
	else
		if (li.type == PointLight) {
			pos = li.vData;
			pA = 1.0f;

			float3 d = sampleUnitSphere(make_float2(sample.y, sample.z));
			dir = d;
			pE = aPdf;
			result += li.emission*M_PIf*4.0f*aPdf;
		}
		else if (li.type == EnvMap) {
			float3 worldCenter = scene_sphere_center;
			float worldRadius = scene_sphere_radius * 1.01f;

			float3 p1 = worldCenter + worldRadius * sampleUnitSphere(make_float2(sample.x, sample.y));
			float3 p2 = worldCenter + worldRadius * sampleUnitSphere(make_float2(sample.z, sample.w));

			pos = p1;
			dir = p2;
			float3 toCenter = normalize(worldCenter - p1);
			float costheta = abs(dot(toCenter, p2));
			if (costheta < 0.00001f)
			{
				costheta = 1.0f;
			}
			pA = costheta;
			pE = aPdf;

			float2 t = ll_map(p2);
			float3 lAttenuation = make_float3(rtTex2D<float4>(1, t.x, t.y));
			result = lAttenuation / (costheta / (4.0f*M_PIf*worldRadius*worldRadius));
		}

		//else

		return result;
}


