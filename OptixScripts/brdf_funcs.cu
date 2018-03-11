
#include <optix.h>
#include <optix_device.h>
#include "shading.h"
#include "simplex_noise.h"
#include "helpers.h"
#include "microfacet.h"

using namespace optix;

rtCallableProgram(float3, color_input, (ShadingState const));

// Only useful for testing nesting calls: Passthrough Color
RT_CALLABLE_PROGRAM float3 color_passthrough(ShadingState const state)
{
	return state.diffuse;
}

RT_CALLABLE_PROGRAM float3 NullSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	return make_float3(0.0f);
}

RT_CALLABLE_PROGRAM float3 NullEval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	return make_float3(0, 0, 0);
}

RT_CALLABLE_PROGRAM float3 LambertEval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Diffuse;

	float3 aLocalDirGen = state.frame.ToLocal(wo);
	bsdfState.pdfW += max(0.f, aLocalDirGen.z * M_1_PIf);
	bsdfState.invPdf += max(0.f, state.IncomingDir.z*M_1_PIf);
	return state.diffuse*M_1_PIf;
}

RT_CALLABLE_PROGRAM float3 LambertSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	float unweightedPdfW;
	float3 oLocalDirGen = SampleCosHemisphereW(make_float2(u0, u1), &unweightedPdfW);

	if (state.IncomingDir.z < EPS_COSINE)
		return make_float3(0.0f);
	val = state.frame.ToWorld(oLocalDirGen);

	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Diffuse;
	bsdfState.pdfW += max(0.f, oLocalDirGen.z * M_1_PIf);
	bsdfState.invPdf += max(0.f, state.IncomingDir.z*M_1_PIf);

	return (state.diffuse * INV_PI_F);
}

RT_CALLABLE_PROGRAM float3 PhongEval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	float3 aLocalDirGen = state.frame.ToLocal(wo);
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 mLocalDirFix = state.IncomingDir;
	const float3 reflLocalDirIn = make_float3(-mLocalDirFix.x, -mLocalDirFix.y, mLocalDirFix.z);
	const float dot_R_Wi = dot(reflLocalDirIn, aLocalDirGen);
	if (dot_R_Wi <= EPS_PHONG)
		return make_float3(0.f);

	const float pdfW = PowerCosHemispherePdfW(reflLocalDirIn, aLocalDirGen, state.phong_exp);
	bsdfState.pdfW += pdfW;
	bsdfState.invPdf += pdfW;

	const float3 rho = state.diffuse *(state.phong_exp + 2.f) * 0.5f * M_1_PIf;

	return rho * pow(dot_R_Wi, state.phong_exp);
}

RT_CALLABLE_PROGRAM float3 PhongSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	float2 aRndTuple = make_float2(u0, u1);
	float3 oLocalDirGen = SamplePowerCosHemisphereW(aRndTuple, state.phong_exp, NULL);
	float3 mLocalDirFix = state.IncomingDir;

	const float3 reflLocalDirFixed = make_float3(-mLocalDirFix.x, -mLocalDirFix.y, mLocalDirFix.z);
	Frame frame(reflLocalDirFixed);
	val = frame.ToWorld(oLocalDirGen);

	const float dot_R_Wi = dot(reflLocalDirFixed, oLocalDirGen);

	if (dot_R_Wi <= EPS_PHONG)
		return make_float3(0.f);
	const float pdfW = PowerCosHemispherePdfW(reflLocalDirFixed, oLocalDirGen, state.phong_exp);

	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;
	bsdfState.pdfW += pdfW;
	bsdfState.invPdf += pdfW;

	const float3 rho = state.diffuse*(state.phong_exp + 2.f) * 0.5f * M_1_PIf;
	return rho * pow(dot_R_Wi, state.phong_exp);
}


RT_CALLABLE_PROGRAM float3 LambPhongEval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	float3 aLocalDirGen = state.frame.ToLocal(wo);
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 mLocalDirFix = state.IncomingDir;
	const float3 reflLocalDirIn = make_float3(-mLocalDirFix.x, -mLocalDirFix.y, mLocalDirFix.z);
	const float dot_R_Wi = dot(reflLocalDirIn, aLocalDirGen);
	if (dot_R_Wi <= EPS_PHONG)
		return make_float3(0.f);

	const float pdfW = PowerCosHemispherePdfW(reflLocalDirIn, aLocalDirGen, state.phong_exp);
	bsdfState.pdfW += pdfW;
	bsdfState.invPdf += pdfW;

	bsdfState.pdfW += max(0.f, aLocalDirGen.z * M_1_PIf);
	bsdfState.invPdf += max(0.f, state.IncomingDir.z*M_1_PIf);

	const float3 rho = state.diffuse *(state.phong_exp + 2.f) * 0.5f * M_1_PIf;

	return state.diffuse * INV_PI_F + rho * pow(dot_R_Wi, state.phong_exp);
}

RT_CALLABLE_PROGRAM float3 LambPhongSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{

	float2 aRndTuple = make_float2(u0, u1);
	float3 oLocalDirGen;
	float3 mLocalDirFix = state.IncomingDir;
	const float3 reflLocalDirFixed = make_float3(-mLocalDirFix.x, -mLocalDirFix.y, mLocalDirFix.z);
	Frame frame(reflLocalDirFixed);

	if (u2 > 0.5f)
	{
		oLocalDirGen = SamplePowerCosHemisphereW(aRndTuple, state.phong_exp, NULL);
		
		val = frame.ToWorld(oLocalDirGen);

		const float dot_R_Wi = dot(reflLocalDirFixed, oLocalDirGen);

		if (dot_R_Wi <= EPS_PHONG)
			return make_float3(0.f);
		const float pdfW = PowerCosHemispherePdfW(reflLocalDirFixed, oLocalDirGen, state.phong_exp);

		bsdfState.cosAt = abs(state.IncomingDir.z);
		bsdfState.bsdfEvent = BsdfEvent_Glossy;
		bsdfState.pdfW += pdfW;
		bsdfState.invPdf += pdfW;

		bsdfState.pdfW += max(0.f, oLocalDirGen.z * M_1_PIf);
		bsdfState.invPdf += max(0.f, state.IncomingDir.z*M_1_PIf);

		const float3 rho = state.diffuse*(state.phong_exp + 2.f) * 0.5f * M_1_PIf;
		return rho * pow(dot_R_Wi, state.phong_exp) + state.diffuse * INV_PI_F;
	}
	else
	{
		float unweightedPdfW;
		oLocalDirGen = SampleCosHemisphereW(make_float2(u0, u1), &unweightedPdfW);

		if (state.IncomingDir.z < EPS_COSINE)
			return make_float3(0.0f);
		const float dot_R_Wi = dot(reflLocalDirFixed, oLocalDirGen);

		val = state.frame.ToWorld(oLocalDirGen);
		const float pdfW = PowerCosHemispherePdfW(reflLocalDirFixed, oLocalDirGen, state.phong_exp);

		bsdfState.cosAt = abs(state.IncomingDir.z);
		bsdfState.bsdfEvent = BsdfEvent_Diffuse;
		bsdfState.pdfW += max(0.f, unweightedPdfW) + pdfW;
		bsdfState.invPdf += max(0.f, state.IncomingDir.z*M_1_PIf) + pdfW;

		const float3 rho = state.diffuse*(state.phong_exp + 2.f) * 0.5f * M_1_PIf;

		return (state.diffuse * INV_PI_F) + rho * pow(dot_R_Wi, state.phong_exp);
	}
}




RT_CALLABLE_PROGRAM float3 MetalSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	float pexp = 7.f;
	float2 aRndTuple = make_float2(u0, u1);
	float pdfW;
	float3 oLocalDirGen = SamplePowerCosHemisphereW(aRndTuple, pexp, &pdfW);
	float3 mLocalDirFix = state.IncomingDir;

	const float3 reflLocalDirFixed = make_float3(-mLocalDirFix.x, -mLocalDirFix.y, mLocalDirFix.z);
	Frame frame(reflLocalDirFixed);
	val = frame.ToWorld(oLocalDirGen);

	const float dot_R_Wi = dot(reflLocalDirFixed, oLocalDirGen);

	if (dot_R_Wi <= EPS_PHONG)
		return make_float3(0.f);

	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Specular;
	bsdfState.pdfW += pdfW;
	bsdfState.invPdf += pdfW;

	const float3 rho = state.diffuse*(pexp + 2.f) * 0.5f * M_1_PIf;
	float3 h = state.IncomingDir + oLocalDirGen;
	float g = G_Smith(state.IncomingDir, oLocalDirGen, state.Ng, min(0.9f, pexp / 100.f));
	//float fr = 	FrCond(state.IncomingDir.z, 1.9f, optix::luminanceCIE(state.diffuse));
	//state.reflectivity;
	float3 Fr = make_float3(FrCond(state.IncomingDir.z, state.ior, state.diffuse.x), FrCond(state.IncomingDir.z, state.ior, state.diffuse.y), FrCond(state.IncomingDir.z, state.ior, state.diffuse.z));

	return rho * pow(dot_R_Wi, pexp)*Fr*g / (4.0f*absdot(state.Ng, state.IncomingDir)*absdot(state.Ng, val));

}

RT_CALLABLE_PROGRAM float3 RefractSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Specular | BsdfEvent_Refraction;
	bsdfState.pdfW += 1.0f;
	bsdfState.invPdf += 1.0f;

	float cosI = state.IncomingDir.z;
	float cosT;
	float etaIncOverEtaTrans;
	if (cosI < 0.f) // hit from inside
	{
		etaIncOverEtaTrans = state.eta;
		cosI = -cosI;
		cosT = 1.f;
	}
	else
	{
		etaIncOverEtaTrans = state.eta;
		cosT = -1.f;
	}
	const float sinI2 = 1.f - cosI * cosI;
	const float sinT2 = (etaIncOverEtaTrans*etaIncOverEtaTrans) * sinI2;

	if (sinT2 < 1.f) // no total internal reflection
	{
		cosT *= sqrtf(max(0.f, 1.f - sinT2));

		val = state.frame.ToWorld(make_float3(
			-etaIncOverEtaTrans * state.IncomingDir.x,
			-etaIncOverEtaTrans * state.IncomingDir.y,
			cosT));
		const float refractCoeff = 1.f - state.reflectivity;
		if (state.adjoint)
			return make_float3(refractCoeff * (etaIncOverEtaTrans*etaIncOverEtaTrans) / abs(cosT));
		else return make_float3(refractCoeff / abs(cosT));
	}
	return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float3 ReflectSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Specular;

	const float albedoReflect = state.reflectivity*optix::luminanceCIE(state.diffuse);

	bsdfState.pdfW += albedoReflect;
	bsdfState.invPdf += albedoReflect;

	float3 oLocalDirGen = make_float3(-state.IncomingDir.x, -state.IncomingDir.y, state.IncomingDir.z);
	val = state.frame.ToWorld(oLocalDirGen);
	return state.reflectivity*state.diffuse;
}

RT_CALLABLE_PROGRAM float3 FresnelGlassSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Specular;
	bsdfState.pdfW += 1.0f;
	bsdfState.invPdf += 1.0f;

	float3 col = isBlack(state.diffuse) ? state.specular : state.diffuse;
	float3 rayDir = state.frame.ToWorld(-state.IncomingDir);
	float3 reflect_dir = reflect(rayDir, state.frame.mZ);
	float3 N = state.frame.mZ;
	float3 i = rayDir;
	float3 t;
	bool into = (dot(rayDir, N) < 0.0f);

	if (refract(t, i, N, state.ior))
	{


		float cos_theta = dot(i, N);
		if (cos_theta < 0.0f)
			cos_theta = -cos_theta;
		else
			cos_theta = dot(t, N);
		float3 color = state.inside ? pow3f(col, -state.distance*0.04f) : col;
		//exp3f(col, -state.distance*0.003f);
		bsdfState.bsdfEvent |= BsdfEvent_Refraction;

		float importance = (1.0f - state.reflectivity);
		float eta = state.eta*state.eta;
		val = t;
		return state.adjoint ? color*importance *eta / abs(cos_theta) : color*importance / abs(cos_theta);
	}
	else{
		//val = state.frame.ToWorld(reflect_dir);
		val = reflect_dir;
		return state.reflectivity*col / abs(state.IncomingDir.z);
	}

}