#pragma once

#include <optix.h>
#include <optix_math.h>

#include "math_utils.h"
#include "commonStructs.h"

#define CALLABLE __device__ __forceinline__


CALLABLE float3 IlluminateAreaLight(
	const Lightsource *const light,
	const float3       &aReceivingPosition,
	const float2       &aRndTuple,
	float3             &oDirectionToLight,
	float             &oDistance,
	float             &oDirectPdfW,
	float             *oEmissionPdfW,
	float             *oCosAtLight)
{
	const float2 uv = SampleUniformTriangle(aRndTuple);
	const float3 e1 = light->triLight.v1 - light->triLight.v2;
	const float3 e2 = light->triLight.v0 - light->triLight.v2;

	const float3 lightPoint = light->triLight.v0 + e1 * uv.x + e2 * uv.y;

	oDirectionToLight = lightPoint - aReceivingPosition;
	const float distSqr = dot(oDirectionToLight, oDirectionToLight);
	oDistance = sqrtf(distSqr);
	oDirectionToLight = oDirectionToLight / oDistance;

	const float cosNormalDir = dot(light->triLight.normal, -oDirectionToLight);
	const float mInvArea = 1.f / length(cross(e1, e2));
	// too close to, or under, tangent
	if (cosNormalDir < EPS_COSINE)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	oDirectPdfW = mInvArea * distSqr / cosNormalDir;
	*oCosAtLight = cosNormalDir;
	*oEmissionPdfW = mInvArea * cosNormalDir * INV_PI_F;

	return light->emission;
}

CALLABLE float3 IlluminateAreaLight(
	const TriangleLight light,
	const float3       &aReceivingPosition,
	const float2       &aRndTuple,
	float3             &oDirectionToLight,
	float             &oDistance,
	float             &oDirectPdfW,
	float             *oEmissionPdfW,
	float             *oCosAtLight)
{
	const float2 uv = SampleUniformTriangle(aRndTuple);
	const float3 e1 = light.v2 - light.v3;
	const float3 e2 = light.v1 - light.v3;

	const float3 lightPoint = light.v1 + e1 * uv.x + e2 * uv.y;

	oDirectionToLight = lightPoint - aReceivingPosition;
	const float distSqr = dot(oDirectionToLight, oDirectionToLight);
	oDistance = sqrtf(distSqr);
	oDirectionToLight = oDirectionToLight / oDistance;

	const float cosNormalDir = dot(light.normal, -oDirectionToLight);
	const float mInvArea = 1.f / length(cross(e1, e2));
	// too close to, or under, tangent
	if (cosNormalDir < EPS_COSINE)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	oDirectPdfW = mInvArea * distSqr / cosNormalDir;
	*oCosAtLight = cosNormalDir;
	*oEmissionPdfW = mInvArea * cosNormalDir * INV_PI_F;

	return light.emission;
}

CALLABLE float3 IlluminateAreaLight(
	const float3	    &v1,
	const float3	    &v2,
	const float3	    &v3,
	const float3        &normal,
	const float3		&emission,
	const float3       &aReceivingPosition,
	const float2       &aRndTuple,
	float3             &oDirectionToLight,
	float             &oDistance,
	float             &oDirectPdfW,
	float             *oEmissionPdfW,
	float             *oCosAtLight)
{
	const float2 uv = SampleUniformTriangle(aRndTuple);
	const float3 e1 = v2 - v3;
	const float3 e2 = v1 - v3;

	const float3 lightPoint = v1 + e1 * uv.x + e2 * uv.y;

	oDirectionToLight = lightPoint - aReceivingPosition;
	const float distSqr = dot(oDirectionToLight, oDirectionToLight);
	oDistance = sqrtf(distSqr);
	oDirectionToLight = oDirectionToLight / oDistance;

	const float cosNormalDir = dot(normal, -oDirectionToLight);
	const float mInvArea = 1.f / length(cross(e1, e2));
	// too close to, or under, tangent
	if (cosNormalDir < EPS_COSINE)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	oDirectPdfW = mInvArea * distSqr / cosNormalDir;
	*oCosAtLight = cosNormalDir;
	*oEmissionPdfW = mInvArea * cosNormalDir * INV_PI_F;

	return emission;
}

CALLABLE float3 EmitAreaLight(
	const float3	  &v0,
	const float3	  &v1,
	const float3	  &v2,
	const float3	  &n,
	const float3	  emission,
	const float2      &aDirRndTuple,
	const float2      &aPosRndTuple,
	float3            &oPosition,
	float3            &oDirection,
	float             &oEmissionPdfW,
	float             *oDirectPdfA,
	float             *oCosThetaLight)
{
	const float2 uv = SampleUniformTriangle(aPosRndTuple);
	const float3 e1 = v1 - v2;
	const float3 e2 = v0 - v2;
	float mInvArea = 1.f / length(cross(e1, e2));
	oPosition = v0 + e1 * uv.x + e2 * uv.y;


	float3 localDirOut = SampleCosHemisphereW(aDirRndTuple, &oEmissionPdfW);

	oEmissionPdfW *= mInvArea;

	// cannot really not emit the particle, so just bias it to the correct angle
	localDirOut.z = max(localDirOut.z, EPS_COSINE);

	float3 V, W;

	createONB(n, V, W);

	oDirection = V * localDirOut.x + W * localDirOut.y + n * localDirOut.z;
	*oDirectPdfA = mInvArea;
	*oCosThetaLight = localDirOut.z;

	return emission * localDirOut.z;
}

CALLABLE float3 GetRadianceAreaLight(
	const Lightsource *const light,
	const float3      &aRayDirection,
	const float3      &aHitPoint,
	float             *oDirectPdfA,
	float             *oEmissionPdfW)
{
	const float cosOutL = max(0.0f, dot(light->triLight.normal, -aRayDirection));

	if (cosOutL == 0)
		return make_float3(0.0f, 0.0f, 0.0f);
	const float3 e1 = light->triLight.v1 - light->triLight.v2;
	const float3 e2 = light->triLight.v0 - light->triLight.v2;
	float mInvArea = 1.f / length(cross(e1, e2));

	*oDirectPdfA = mInvArea;
	*oEmissionPdfW = CosHemispherePdfW(light->triLight.normal, -aRayDirection) * mInvArea;

	return light->emission;
}

CALLABLE float3 GetRadianceAreaLight(
	const TriangleLight *const light,
	const float3      &aRayDirection,
	const float3      &aHitPoint,
	float             *oDirectPdfA,
	float             *oEmissionPdfW)
{
	const float cosOutL = max(0.0f, dot(light->normal, -aRayDirection));

	if (cosOutL == 0)
		return make_float3(0.0f, 0.0f, 0.0f);
	const float3 e1 = light->v2 - light->v3;
	const float3 e2 = light->v1 - light->v3;
	float mInvArea = 1.f / length(cross(e1, e2));

	*oDirectPdfA = mInvArea;
	*oEmissionPdfW = CosHemispherePdfW(light->normal, -aRayDirection) * mInvArea;

	return light->emission;
}


CALLABLE float3 GetRadianceAreaLight(
	const float3      &v1,
	const float3      &v2,
	const float3      &v3,
	const float3      &n,
	const float3      &emission,
	const float3      &aRayDirection,
	const float3      &aHitPoint,
	float             *oDirectPdfA,
	float             *oEmissionPdfW)
{
	const float cosOutL = max(0.0f, dot(n, -aRayDirection));

	if (cosOutL == 0)
		return make_float3(0.0f, 0.0f, 0.0f);
	const float3 e1 = v2 - v3;
	const float3 e2 = v1 - v3;
	float mInvArea = 1.f / length(cross(e1, e2));

	*oDirectPdfA = mInvArea;
	*oEmissionPdfW = CosHemispherePdfW(n, -aRayDirection) * mInvArea;

	return emission;
}


CALLABLE float3 GetRadianceBackground(
	const rtTextureId&  mTex,
	const float		   mInvSceneRadiusSqr,
	const float3       &aRayDirection,
	const float3       &aHitPoint,
	float             *oDirectPdfA,
	float             *oEmissionPdfW)
{
	float directPdf = UniformSpherePdfW();
	float2 t = ll_map(normalize(aRayDirection));
	float3 lAttenuation = abs(make_float3(rtTex2D<float4>(mTex, t.x, t.y)));
	const float positionPdf = ConcentricDiscPdfA() *mInvSceneRadiusSqr;

	if (oDirectPdfA)
		*oDirectPdfA = directPdf;

	if (oEmissionPdfW)
		*oEmissionPdfW = directPdf * positionPdf;

	return lAttenuation;
}


CALLABLE float3 IlluminateBackground(
	const rtTextureId&  mTex,
	const float		   mInvSceneRadiusSqr,
	const float3       &aReceivingPosition,
	const float2       &aRndTuple,
	float3             &oDirectionToLight,
	float              &oDistance,
	float              &oDirectPdfW,
	float              *oEmissionPdfW,
	float              *oCosAtLight)
{
	// Replace these two lines with image sampling
	oDirectionToLight = SampleUniformSphereW(aRndTuple, &oDirectPdfW);

	// This stays even with image sampling
	oDistance = 1e36f;
	*oEmissionPdfW = oDirectPdfW * ConcentricDiscPdfA() * mInvSceneRadiusSqr;
	*oCosAtLight = 1.f;
	float2 t = ll_map(normalize(oDirectionToLight));
	float3 lAttenuation = abs(make_float3(rtTex2D<float4>(mTex, t.x, t.y)));

	return lAttenuation;
}

/*
class BaseLightsource
{
public:
float3 Power;
int    Flags;

virtual __device__ float3 Eval(const float3& dir, float& pdf)
{
return Power;
}

virtual __device__ float3 Illuminate(const float3& pos, const float3& n, const float3& wi, const float4& sample,
float3* lightDir, float* cosAt, float* emissionPdf, float* areaPdf)
{
return make_float3(0.f);
}

virtual __device__ float3 Emit(const float4& sample, float sample1, float3* pos, float3* lightDir, float* cosAt, float* emissionPdf, float* areaPdf)
{
return make_float3(0.f);
}
};

class Pointlight : public BaseLightsource
{
public:
float3 Position;

__device__ Pointlight(const float3& p, const float3& pw)
{
Power = pw;
Position = p;
}

__device__ float3 Illuminate(const float3& pos, const float3& n, const float3& wi, const float4& sample,
float3* lightDir, float* cosAt, float* emissionPdf, float* areaPdf)
{


float3 L = (Position - pos);
float Ldist = length(L);
float nmDl = dot(n, normalize(L));
if (nmDl > 0.0f)
{
*lightDir = L;
float3 lAttenuation = Power;
*areaPdf = 1.0f / Ldist*Ldist;
*emissionPdf *= 4*M_PIf;
return (lAttenuation) ;
}

return make_float3(0.f);
};
};

class TriangleLightsource : public BaseLightsource
{
public:
float3 v0, v1, v2, n;


};*/