#pragma once

#include <optix_math.h>
using namespace optix;

#define EPS_COSINE 1e-6f
#define PI_F     3.1415926535897932384626433832795f
#define INV_PI_F 0.31830988618379067153776752674503f
#define EPS_PHONG 1e-3f

#define isZero(v) (length(v) == 0)
#define isBlack(v) (length(v) == 0 || isnan(length(v)))



struct Frame
{
public:
	float3 mX, mY, mZ;

	__device__ Frame()
	{
		mX = make_float3(1.f, 0.f, 0.f);
		mY = make_float3(0.f, 1.f, 0.f);
		mZ = make_float3(0.f, 0.f, 1.f);
	}

	__device__ Frame(const float3& z)
	{
		float3 c = mZ = normalize(z);
		float3 tmpX = (abs(c.x) > 0.99f) ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
		mY = normalize(cross(c, tmpX));
		mX = cross(mY, c);
	}

	__device__ float3 ToWorld(const float3& a) const
	{
		return mX * a.x + mY * a.y + mZ * a.z;
	}

	__device__ float3 ToLocal(const float3& a) const
	{
		return make_float3(dot(a, mX), dot(a, mY), dot(a, mZ));
	}
};


__device__ __forceinline__ float3 SamplePowerCosHemisphereW(
	const float2  &aSamples,
	const float  aPower,
	float        *oPdfW)
{
	const float term1 = 2.f * PI_F * aSamples.x;
	const float term2 = pow(aSamples.y, 1.f / (aPower + 1.f));
	const float term3 = sqrt(1.f - term2 * term2);

	if (oPdfW)
	{
		*oPdfW = (aPower + 1.f) * pow(term2, aPower) * (0.5f * M_1_PIf);
	}

	return make_float3(
		cos(term1) * term3,
		sin(term1) * term3,
		term2);
}

__device__ __forceinline__ float PowerCosHemispherePdfW(
	const float3  &aNormal,
	const float3&aDirection,
	const float  aPower)
{
	const float cosTheta = max(0.f, dot(aNormal, aDirection));

	return (aPower + 1.f) * pow(cosTheta, aPower) * (M_1_PIf * 0.5f);
}

__device__ __forceinline__ float3 SampleCosHemisphereW(
	const float2  &aSamples,
	float        *oPdfW)
{
	const float term1 = 2.f * PI_F * aSamples.x;
	const float term2 = sqrtf(1.f - aSamples.y);

	const float3 ret = make_float3(
		cosf(term1) * term2,
		sinf(term1) * term2,
		sqrtf(aSamples.y));

	if (oPdfW)
	{
		*oPdfW = ret.z * INV_PI_F;
	}

	return ret;
}

//////////////////////////////////////////////////////////////////////////
// Utilities for converting PDF between Area (A) and Solid angle (W)
// WtoA = PdfW * cosine / distance_squared
// AtoW = PdfA * distance_squared / cosine

__device__ __forceinline__ float PdfWtoA(
	const float aPdfW,
	const float aDist,
	const float aCosThere)
{
	return aPdfW * abs(aCosThere) / (aDist * aDist);
}

//////////////////////////////////////////////////////////////////////////
// Disc sampling

__device__ __forceinline__ float2 SampleConcentricDisc(
	const float2 &aSamples)
{
	float phi, r;

	float a = 2.f * aSamples.x - 1.f;   /* (a,b) is now on [-1,1]^2 */
	float b = 2.f * aSamples.y - 1.f;

	if (a > -b)      /* region 1 or 2 */
	{
		if (a > b)   /* region 1, also |a| > |b| */
		{
			r = a;
			phi = (PI_F / 4.f) * (b / a);
		}
		else        /* region 2, also |b| > |a| */
		{
			r = b;
			phi = (PI_F / 4.f) * (2.f - (a / b));
		}
	}
	else            /* region 3 or 4 */
	{
		if (a < b)   /* region 3, also |a| >= |b|, a != 0 */
		{
			r = -a;
			phi = (PI_F / 4.f) * (4.f + (b / a));
		}
		else        /* region 4, |b| >= |a|, but a==0 and b==0 could occur. */
		{
			r = -b;

			if (b != 0)
				phi = (PI_F / 4.f) * (6.f - (a / b));
			else
				phi = 0;
		}
	}

	float2 res;
	res.x = r * cosf(phi);
	res.y = r * sinf(phi);
	return res;
}

__device__ __forceinline__ float3 SampleUniformSphereW(
	const float2  &aSamples,
	float        *oPdfSA)
{
	const float term1 = 2.f * PI_F * aSamples.x;
	const float term2 = 2.f * sqrtf(aSamples.y - aSamples.y * aSamples.y);

	const float3 ret = make_float3(
		cosf(term1) * term2,
		sinf(term1) * term2,
		1.f - 2.f * aSamples.y);

	if (oPdfSA)
	{
		*oPdfSA = INV_PI_F * 0.25f;
	}

	return ret;
}

__device__ __forceinline__ float CosHemispherePdfW(
	const float3  &aNormal,
	const float3  &aDirection)
{
	float val = dot(aNormal, aDirection);
	if (val < 0.0f)
		val = 0.0f;
	return val * INV_PI_F;
}

// Sample Triangle
// returns barycentric coordinates
__device__ __forceinline__ const float2 SampleUniformTriangle(const float2 &aSamples)
{
	const float term = sqrtf(aSamples.x);

	return make_float2(1.f - term, aSamples.y * term);
}

__device__ __host__ __forceinline__ float UniformSpherePdfW()
{
	return INV_PI_F * 0.25f;
}

__device__ __host__ __forceinline__ float ConcentricDiscPdfA()
{
	return INV_PI_F;
}

__device__ __forceinline__ float PdfAtoW(
	const float aPdfA,
	const float aDist,
	const float aCosThere)
{
	return aPdfA * (aDist * aDist) / abs(aCosThere);
}

// Mis power (1 for balance heuristic)
__device__ __forceinline__ float Mis(const float aPdf)
{
	return aPdf;
}

// Mis weight for 2 pdfs
__device__ __forceinline__ float Mis2(
	const float aSamplePdf,
	const float aOtherPdf)
{
	return Mis(aSamplePdf) / (Mis(aSamplePdf) + Mis(aOtherPdf));
}

__device__ __forceinline__ float maximum(const float3 &v)
{
	return max(v.x, max(v.y, v.z));
}

__device__ __forceinline__ float FresnelDielectric(
	float aCosInc,
	const float mIOR)
{
	if (mIOR < 0.f)
		return 1.f;

	float etaIncOverEtaTrans;

	if (aCosInc < 0.f)
	{
		aCosInc = -aCosInc;
		etaIncOverEtaTrans = mIOR;
	}
	else
	{
		etaIncOverEtaTrans = 1.f / mIOR;
	}

	const float sinTrans2 = (etaIncOverEtaTrans * etaIncOverEtaTrans) * (1.f - (aCosInc * aCosInc));
	const float cosTrans = sqrtf(max(0.f, 1.f - sinTrans2));

	const float term1 = etaIncOverEtaTrans * cosTrans;
	const float rParallel =
		(aCosInc - term1) / (aCosInc + term1);

	const float term2 = etaIncOverEtaTrans * aCosInc;
	const float rPerpendicular =
		(term2 - cosTrans) / (term2 + cosTrans);

	return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}


__device__ __forceinline__ float3 SphericalDir(const float sintheta, const float costheta, const float phi)
{
	return make_float3(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
}

__device__ __forceinline__ float sqr(const float c)
{
	return c * c;
}

__device__ __inline__ float2 ll_map(const float3& dir)
{
	float3 direction = normalize(dir);
	float theta = atan2f(direction.x, direction.y);
	float phi = M_PIf * 0.5f - acosf(direction.z);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sinf(phi));
	return make_float2(u, v);
}
