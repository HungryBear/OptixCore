#include <optix.h>
#include "shading.h"
#include "helpers.h"
#include "microfacet.h"


rtCallableProgram(float3, sampleProgram1, (ShadingState const, BsdfState&, float, float, float, float3&));
rtCallableProgram(float3, sampleProgram2, (ShadingState const, BsdfState&, float, float, float, float3&));


RT_CALLABLE_PROGRAM float3 Layer2Sample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	if (u2 >= 0.5f)
	{
		return 0.5f*sampleProgram1(state, bsdfState, u0, u1, u2, val);
	}
	else
		return 0.5f*sampleProgram2(state, bsdfState, u0, u1, u2, val);
}

rtCallableProgram(float3, evalProgram1, (ShadingState const, BsdfState&, float3));
rtCallableProgram(float3, evalProgram2, (ShadingState const, BsdfState&, float3));
rtCallableProgram(float3, layerProc, (ShadingState const, float3, float3));


RT_CALLABLE_PROGRAM float3 Layer2Eval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	float3 e_a = evalProgram1(state, bsdfState, wo);
	float3 e_b = evalProgram2(state, bsdfState, wo);

	return layerProc(state, e_a, e_b);
}
rtDeclareVariable(float, layer_Height, , ) = 0.7f;
rtDeclareVariable(float, layer_ior, , ) = 1.4f;
rtDeclareVariable(float, layer_absorb, , ) = 0.7f;


RT_CALLABLE_PROGRAM float3 Layer2EvalDiel(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	float3 e_a = evalProgram1(state, bsdfState, wo);
	float3 wi_0, wo_0;
	float3 N = state.frame.ToLocal(state.frame.mZ);
	float3 o = state.frame.ToLocal(wo);
	if (refract(wi_0, state.IncomingDir, N, state.ior))
	{
		refract(wo_0, wo, N, state.ior);

		float Fr12 = FrDiel(
			dot(state.IncomingDir, N),
			dot(o, N),
			state.ior / layer_ior, layer_ior / state.ior);

		float Fr21 = FrDiel(
			dot(o, N),
			dot(state.IncomingDir, N),
			layer_ior / state.ior,
			state.ior / layer_ior);

		ShadingState shadingState;
		shadingState.uv = state.uv;
		shadingState.Ng = state.Ng;
		shadingState.adjoint = state.adjoint;
		shadingState.frame = state.frame;
		shadingState.distance = layer_Height;

		float dist = layer_Height*(
			(1.f / acosf(dot(N, wo_0))) *
			(1.f / acosf(dot(N, wi_0)))
			);

		shadingState.seed = state.seed;
		shadingState.phong_exp = state.phong_exp;
		shadingState.eta = layer_ior / state.ior;
		shadingState.ior = layer_ior;
		shadingState.IncomingDir = shadingState.frame.ToLocal(-wi_0);
		shadingState.reflectivity = FresnelDielectric(shadingState.IncomingDir.z, layer_ior);
		shadingState.diffuse = state.diffuse;
		shadingState.specular = state.specular;
		shadingState.inside = state.inside;

		float3 e_b = evalProgram2(shadingState, bsdfState, -wo_0);

		float a = expf(-layer_absorb*dist);
		float t = 1.0f;
		return e_a + Fr12*e_b*a*Fr21;
	}
	else
	{
		return e_a;
	}

};
