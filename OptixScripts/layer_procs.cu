#include <optix_world.h>
#include "shading.h"
#include "simplex_noise.h"
#include "helpers.h"


RT_CALLABLE_PROGRAM float3 LayerAdd(ShadingState const state, const float3 a, const float3 b)
{
	return a + b;
}

RT_CALLABLE_PROGRAM float3 LayerLerp(ShadingState const state, const float3 a, const float3 b)
{	
	float wt = 0.5f;
	return make_float3(lerp(a.x, b.x, wt), lerp(a.y, b.y, wt), lerp(a.z, b.z, wt));
}

RT_CALLABLE_PROGRAM float3 LayerLerpCos(ShadingState const state, const float3 a, const float3 b)
{
	float wt = abs(state.IncomingDir.z);
	return make_float3(lerp(a.x, b.x, wt), lerp(a.y, b.y, wt), lerp(a.z, b.z, wt));
}

RT_CALLABLE_PROGRAM float3 LerpSnoise(ShadingState const state, const float3 a, const float3 b)
{
	float wt = snoise(state.uv);
	return make_float3(lerp(a.x, b.x, wt), lerp(a.y, b.y, wt), lerp(a.z, b.z, wt));
}

RT_CALLABLE_PROGRAM float3 LayerFresnelDiel(ShadingState const state, const float3 a, const float3 b)
{
	float wt = state.reflectivity;
	return make_float3(lerp(a.x, b.x, wt), lerp(a.y, b.y, wt), lerp(a.z, b.z, wt));
}

rtDeclareVariable(int, mask_sampler, , );

RT_CALLABLE_PROGRAM float3 LayerMask(ShadingState const state, const float3 a, const float3 b)
{
	float3 maskValue = make_float3(rtTex2D<float4>(mask_sampler, state.uv.x, state.uv.y));
	
	return a*maskValue + (make_float3(1.0f) - maskValue)*b;
}

