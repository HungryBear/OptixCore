#include <optix.h>
#include <optix_world.h>
#include "Shading.h"
#include "helpers.h"
#include "Microfacet.h"


__device__ __inline__ float eval_reflect(ShadingState const state, const float3 wiW, float& pdf)
{
	float3 m_N = state.frame.ToLocal(state.frame.mZ);

	float m_ag = state.GetRoughness();
	float3 wo = state.IncomingDir;
	float3 wi = state.frame.ToLocal(wiW);
	float cosNO = dot(m_N, wo);
	float cosNI = dot(m_N, wi);
	if (cosNI > 0.f && cosNO > 0.f) {
		// get half vector
		float3 Hr = normalize(wi + wo);

		float alpha2 = m_ag * m_ag;
		float cosThetaM = dot(m_N, Hr);
		float cosThetaM2 = cosThetaM * cosThetaM;
		float tanThetaM2 = (1.f - cosThetaM2) / cosThetaM2;
		float cosThetaM4 = cosThetaM2 * cosThetaM2;
		float D = alpha2 / (M_PIf * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

		float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
		float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));
		float G = G1o * G1i;
		float out = (G * D) * 0.25f / cosNO;
		float pm = D * cosThetaM;

		pdf = pm * 0.25f / dot(Hr, wo);
		return out;
	}

	return 0.0f;
}

__device__ __inline__ float eval_refract(ShadingState const state, const float3 wiW, float& pdf)
{
	float3 m_N = state.frame.ToLocal(state.frame.mZ);
	float m_ag = state.GetRoughness();
	float m_eta = state.eta;
	float3 wo = state.IncomingDir;
	float3 wi = state.frame.ToLocal(wiW);
	float cosNO = dot(m_N, wo);
	float cosNI = dot(m_N, wi);
	if (cosNO <= 0.f || cosNI >= 0.f)
		return 0.f;
	float3 ht = -(m_eta * wi + wo);
	float3 Ht = normalize(ht);
	float cosHO = dot(Ht, wo);

	float cosHI = dot(Ht, wi);
	// eq. 33: first we calculate D(m) with m=Ht:
	float alpha2 = m_ag * m_ag;
	float cosThetaM = dot(m_N, Ht);
	float cosThetaM2 = cosThetaM * cosThetaM;
	float tanThetaM2 = (1.f - cosThetaM2) / cosThetaM2;
	float cosThetaM4 = cosThetaM2 * cosThetaM2;
	float D = alpha2 / (M_PIf * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
	// eq. 34: now calculate G1(i,m) and G1(o,m)
	float G1o = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNO * cosNO) / (cosNO * cosNO)));
	float G1i = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNI * cosNI) / (cosNI * cosNI)));
	float G = G1o * G1i;
	// probability
	float invHt2 = 1.f / dot(ht, ht);
	pdf = D * fabsf(cosThetaM) * (fabsf(cosHI) * (m_eta * m_eta)) * invHt2;
	return  (fabsf(cosHI * cosHO) * (m_eta * m_eta) * (G * D) * invHt2) / cosNO;
}

RT_CALLABLE_PROGRAM float3 BlenderMicrofacetEval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	float alpha = state.GetRoughness();
	float3 aLocalDirGen = state.frame.ToLocal(wo);

	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;
	bool reflect = state.IncomingDir.z > aLocalDirGen.z;
	float pdf;
	float3 result = make_float3(1.0f);

	if (reflect)
	{
		result *= eval_reflect(state, wo, pdf);
	}
	else
	{
		result *= eval_refract(state, wo, pdf);
	}
	bsdfState.pdfW += pdf;
	bsdfState.invPdf += pdf;

	return result;

}

RT_CALLABLE_PROGRAM float3 BlenderMicrofacetSample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy; // TODO if a small -> Specular

	float3 omega_in, result;
	float pdfW;
	float3 m_N = state.frame.ToLocal(state.frame.mZ); //!!
	float3 omega_out = state.IncomingDir;
	float m_ag = state.GetRoughness();
	float m_eta = state.eta;
	float cosNO = dot(m_N, omega_out);

	if (cosNO > 0.f)
	{
		float alpha2 = m_ag * m_ag;
		float tanThetaM2 = alpha2 * u0 / (1.f - u0);
		float cosThetaM = 1.f / sqrtf(1 + tanThetaM2);
		float sinThetaM = cosThetaM * sqrtf(tanThetaM2);
		float phiM = 2.f * M_PIf * u1;
		float3 m = make_float3(cosf(phiM) * sinThetaM, (sinf(phiM) * sinThetaM), cosThetaM);

		//float3 m = state.frame.ToWorld(lm);
		float3 color = state.inside ? exp3f(state.diffuse, -state.distance*0.001f) : make_float3(1.f);

		float Fr0 = FresnelDielectric(dot(state.IncomingDir, m), state.ior);
		if (u2 <= Fr0)
		{
			float cosMO = dot(m, omega_out);
			if (cosMO > 0.0f)
			{
				omega_in = 2.f * cosMO * m - omega_out;
				if (dot(state.frame.ToLocal(state.Ng), omega_in) > 0.f)
				{
					float cosThetaM2 = cosThetaM * cosThetaM;
					float cosThetaM4 = cosThetaM2 * cosThetaM2;
					float D = alpha2 / (M_PIf * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
					// eq. 24
					float pm = D * cosThetaM;

					pdfW = pm * 0.25f / cosMO;
					// Eval BRDF*cosNI
					float cosNI = dot(m_N, omega_in);
					float G1o = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNO * cosNO) / (cosNO * cosNO)));
					float G1i = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNI * cosNI) / (cosNI * cosNI)));
					float G = G1o * G1i;
					// eq. 20: (F*G*D)/(4*in*on)
					float out = (G * D) * Fr0 / cosNO;
					result = color * out;
				}
			}
		}
		else
		{
			if (refract(omega_in, state.IncomingDir, m, state.ior))
			{

				bsdfState.bsdfEvent |= BsdfEvent_Refraction;

				float cosThetaM2 = cosThetaM * cosThetaM;
				float cosThetaM4 = cosThetaM2 * cosThetaM2;
				float D = alpha2 / ((M_PIf)* cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
				// eq. 24
				float pm = D * cosThetaM;
				// eval BRDF*cosNI
				float cosNI = dot(m_N, omega_in);

				float G1o = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNO * cosNO) / (cosNO * cosNO)));
				float G1i = 2.f / (1.f + sqrtf(1.f + alpha2 * (1.f - cosNI * cosNI) / (cosNI * cosNI)));
				float G = G1o * G1i;
				// eq. 21
				float cosHI = dot(m, omega_in);
				float cosHO = dot(m, omega_out);
				float Ht2 = m_eta * cosHI + cosHO;
				Ht2 *= Ht2;
				float out = (1.0f - Fr0)*(fabsf(cosHI * cosHO) * (m_eta * m_eta) * (G * D)) / (cosNO * Ht2);

				result = make_float3(out);
				// eq. 38 and eq. 17
				pdfW = pm * (m_eta * m_eta) * fabsf(cosHI) / Ht2;
			}
		}

	}
	else
		return make_float3(0.0f);

	bsdfState.pdfW += pdfW;
	//bsdfState.invPdf += pdfW;
	val = state.frame.ToWorld(omega_in);

	return result;
}


__device__ __inline__ float Blinn_D(const float3 wh, const float exponent)
{
	float ch = abs(wh.z);
	return (exponent + 2.f)*(1.0f / (2.0f*M_PIf))*powf(ch, exponent);
}

__device__ __inline__ float Blinn_Pdf(const float3 wi, const float3 wo, const float exponent)
{
	float3 wh = normalize(wi + wo);
	float cost = abs(wh.z);

	float pdf = ((exponent + 1.f)*powf(cost, exponent)) / (2.f*M_PIf*4.0f*dot(wo, wh));
	if (dot(wo, wh) <= 0.f)
		pdf = 0.f;
	return pdf;
}

__device__ __inline__ float D_GTR(const float3 wh, const float alpha, const float gamma = 1.9f)
{
	const float c = 2.0f;
	float cost2 = abs(wh.z)*abs(wh.z);
	float sint2 = 1.0f - cost2;

	return c / powf(alpha*alpha*cost2 + sint2, gamma);
}

__device__ __inline__ float D_GGX(const float3 wh, const float3 n, const float a)
{
	float cost = wh.z;
	if (cost <= 0)
		return 0.f;
	/*float t = tanf(acosf(cost));
	return (a*a) / (M_PIf*sqr(sqr(cost))*(a*a + t*t));*/
	float t = tanf(acosf(cost));
	return 1.0f / (a*a*M_PIf* sqr(sqr(cost)) *sqr(t*t / a * a));
}

__device__ __inline__ float G1_GGX(const float3 v, const float3 m, const float alpha)
{
	if ((dot(v, m)) <= 0)
		return 0.0f;
	float g = (-1.f + sqrtf(1.f + (1.f / sqr(alpha)))) / 2.f;
	return 1.f / (1.f + g);
	//2.f / (1.f + sqrtf(1.f + sqr(alpha*tanf(acosf(dot(v, m))))));
}

__device__ __inline__ float G_GGX(const float3& i, const float3& o, const float3& m, const float a)
{
	return G1_GGX(i, m, a) * G1_GGX(o, m, a);
}

__device__ __inline__ float GGX_Pdf(const float3 wo, const float3 wi, const float3 n, const float alpha)
{
	float3 wh = normalize(wi + wo);
	float cost = abs(wh.z);

	float  g_pdf = G_GGX(wi, wo, wh, alpha);

	return (abs(dot(wi, wh))*g_pdf) / (abs(dot(wi, n)) * abs(dot(wh, n)));
}

__device__ __inline__ float D_Beckmann(const float cosThetaM, const float alpha)
{
	if (cosThetaM <= 0.0f)
		return 0.0f;
	float M = alpha * alpha;
	float T = cosThetaM * cosThetaM;
	return expf((T - 1.0f) / (M*T)) / (M_PIf*M*T*T);
}

__device__ __inline__ float G1_Beckmann(const float3 v, const float3 m, const float alpha)
{
	if ((dot(v, m)) <= 0)
		return 0.0f;
	const float a = 1.0f / alpha * tanf(acosf(dot(v, m)));
	if (a >= 1.6f)
		return 1.0f;
	return (3.535f*a + 2.181f*a*a) / (1.0f + 2.276f*a + 2.257*a*a);
}

__device__ __inline__ float G_Beckmann(const float3& i, const float3& o, const float3& m, const float a)
{
	return G1_Beckmann(i, m, a) * G1_Beckmann(o, m, a);
}

__device__ __inline__ float Beckmann_Pdf(const float3 wo, const float3 wi, const float3 n, const float alpha)
{
	float3 wh = normalize(wi + wo);
	float cost = abs(wh.z);

	float  g_pdf = G_Beckmann(wi, wo, wh, alpha);

	return (abs(dot(wi, wh))*g_pdf) / (abs(dot(wi, n)) * abs(dot(wh, n)));
}

__device__ __inline__ float G_Microfacet(const float3 wo, const float3 wi, const float3 wh)
{
	const float ndotwh = abs(wh.z);
	const float ndotwo = abs(wo.z);
	const float ndotwi = abs(wi.z);
	float wdotwh = dot(wo, wh);
	return min(1.f, min(
		(2.f*ndotwh*ndotwo / wdotwh)
		,
		(2.f*ndotwh*ndotwi / wdotwh)
	));
}

__device__ __inline__ float GetAlpha(ShadingState const state)
{
	float a = state.GetRoughness()*(state.phong_exp*0.01f);//
	return sqr(0.5f + a / 2.0f);
}

__device__ __inline__ float GetAlpha_GGX(ShadingState const state)
{
	float a = state.roughness;//
							  //return a;
	return sqr(0.5f + a / 2.0f);
}

__device__ __inline__ float EvalBrdf(ShadingState const state, const float3 wi, const float3 wo)
{
	float cost_o = abs(wo.z);
	float cost_i = abs(wi.z);

	if (cost_o < EPS_COSINE && cost_i < EPS_COSINE)
		return 0.f;
	float3 wh = normalize(wi + wo);
	float cost_h = dot(wi, wh);
	float F = FresnelDielectric(cost_h, state.ior);
	float G = G_Microfacet(wo, wi, wh);
	//G_Smith(wi, wo, wh, state.GetRoughness()); //
	float D = Blinn_D(wh, state.phong_exp);


	return F * D*G / (4.f * cost_o*cost_i);
}

__device__ __inline__ float CT_EvalBrdf(ShadingState const state, const float3 wi, const float3 wo)
{
	float cost_o = abs(wo.z);
	float cost_i = abs(wi.z);

	float alpha = GetAlpha(state);

	if (cost_o < EPS_COSINE && cost_i < EPS_COSINE)
		return 0.f;
	float3 wh = normalize(wi + wo);
	float cost_h = dot(wi, wh);
	float F = FresnelDielectric(cost_h, state.ior);
	float G = G_Beckmann(wi, wo, wh, alpha);
	float D = D_Beckmann(abs(wh.z), alpha);

	return (F*D*G) / (4.f * cost_o*cost_i);
}

__device__ __inline__ float GGX_EvalBrdf(ShadingState const state, const float3 wi, const float3 wo)
{
	float cost_o = abs(wo.z);
	float cost_i = abs(wi.z);

	float alpha = GetAlpha_GGX(state);

	if (cost_o < EPS_COSINE && cost_i < EPS_COSINE)
		return 0.f;
	float3 wh = normalize(wi + wo);
	float cost_h = dot(wi, wh);

	float F = FrDiel(cost_i, cost_o, state.incident_ior, state.ior);
	float G = G_GGX(wi, wo, wh, alpha);
	float D = D_GGX(wh, state.frame.mZ, alpha);

	return (F*D*G) / (4.f * cost_o*cost_i);
}

__device__ __inline__ float3 GGX_EvalBrdf_Conductor(ShadingState const state, const float3 wi, const float3 wo)
{
	float cost_o = abs(wo.z);
	float cost_i = abs(wi.z);

	float alpha = GetAlpha_GGX(state);

	if (cost_o < EPS_COSINE && cost_i < EPS_COSINE)
		return make_float3(0.f);
	float3 wh = normalize(wi + wo);
	float cost_h = dot(wi, wh);

	float F = FrCond(cost_h, state.ior, optix::luminanceCIE(state.diffuse));

	float G = G_GGX(wi, wo, wh, alpha);
	float D = D_GGX(wh, state.frame.mZ, alpha);

	return state.diffuse*(F*D*G) / (4.f * cost_o*cost_i);
}

__device__ __inline__ float GGX_EvalBrdf_Refract(ShadingState const state, const float3 wi, const float3 wo)
{
	float no = state.ior;
	float ni = state.incident_ior;

	const float3 n = state.frame.mZ;

	float alpha = GetAlpha_GGX(state);
	float cost_o = abs(wo.z);
	float cost_i = abs(wi.z);

	float3 wh = normalize(wi + wo);
	float cost_h = dot(wi, wh);

	float F = FrDiel(cost_i, cost_o, state.incident_ior, state.ior);
	float G = G_GGX(wi, wo, wh, alpha);
	float D = D_GGX(wh, n, alpha);

	float vc = (absdot(wi, wh)*absdot(wo, wh)) / (absdot(n, wi)*absdot(n, wi));
	if (isnan(vc))
	{
		return 0.f;
	}

	return vc * ((no*no*(1.f - F)*G*D) / sqr(ni*dot(wi, wh) + no * dot(wh, wo)));
}
RT_CALLABLE_PROGRAM float3 BlinnPhong_Microfacet_Eval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 lwo = state.frame.ToLocal(wo);

	bsdfState.pdfW += Blinn_Pdf(lwo, state.IncomingDir, state.phong_exp);
	bsdfState.invPdf += Blinn_Pdf(state.IncomingDir, lwo, state.phong_exp);

	return state.diffuse*EvalBrdf(state, state.IncomingDir, lwo);
}

RT_CALLABLE_PROGRAM float3 BlinnPhong_Microfacet_Sample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 wo = state.IncomingDir;
	float3 wi;

	float costheta = powf(u0, 1.f / (state.phong_exp + 1.f));
	float sintheta = sqrtf(max(0.f, 1.f - costheta * costheta));
	float phi = u1 * 2.f*M_PIf;

	float3 wh = SphericalDir(sintheta, costheta, phi);

	if (!(wo.z*wh.z > 0.f))
	{
		wh = -wh;
	}
	wi = -wo + 2.f*dot(wo, wh)*wh;

	val = state.frame.ToWorld(wi);

	bsdfState.pdfW += Blinn_Pdf(wi, wo, state.phong_exp);
	bsdfState.invPdf += Blinn_Pdf(wo, wi, state.phong_exp);

	return state.diffuse*EvalBrdf(state, wo, wi);

}

RT_CALLABLE_PROGRAM float3 CT_Beckmann_Microfacet_Eval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 lwo = state.frame.ToLocal(wo);

	bsdfState.pdfW += Beckmann_Pdf(lwo, state.IncomingDir, state.frame.mZ, state.phong_exp);
	bsdfState.invPdf += Beckmann_Pdf(state.IncomingDir, lwo, state.frame.mZ, state.phong_exp);

	return state.diffuse*abs(CT_EvalBrdf(state, state.IncomingDir, lwo));
}

RT_CALLABLE_PROGRAM float3 CT_Beckmann_Microfacet_Sample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 wo = state.IncomingDir;
	float3 wi;

	float alpha = GetAlpha(state);
	float theta = atanf(sqrtf(alpha*alpha*logf(1.0f - u0)));
	float phi = u1 * 2.f*M_PIf;

	float3 wh = SphericalDir(sinf(theta), cosf(theta), phi);

	if (!(wo.z*wh.z > 0.f))
	{
		wh = -wh;
	}
	wi = -wo + 2.f*dot(wo, wh)*wh;

	val = state.frame.ToWorld(wi);

	bsdfState.pdfW += Beckmann_Pdf(wi, wo, state.frame.mZ, state.phong_exp);
	bsdfState.invPdf += Beckmann_Pdf(wo, wi, state.frame.mZ, state.phong_exp);

	return state.diffuse*abs(CT_EvalBrdf(state, wi, wo));

}

RT_CALLABLE_PROGRAM float3 CT_GGX_Microfacet_Eval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 lwo = state.frame.ToLocal(wo);

	bsdfState.pdfW += GGX_Pdf(lwo, state.IncomingDir, state.frame.mZ, GetAlpha(state));
	bsdfState.invPdf += GGX_Pdf(state.IncomingDir, lwo, state.frame.mZ, GetAlpha(state));


	return state.diffuse*(GGX_EvalBrdf(state, state.IncomingDir, lwo) + GGX_EvalBrdf_Refract(state, state.IncomingDir, lwo));
}

RT_CALLABLE_PROGRAM float3 CT_GGX_Microfacet_Sample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 wo = state.IncomingDir;
	float3 wi;

	float alpha = GetAlpha_GGX(state);
	float theta = atanf(sqrtf(alpha*alpha*logf(1.0f - u0)));
	float phi = u1 * 2.f*M_PIf;

	float3 wh = SphericalDir(sinf(theta), cosf(theta), phi);
	float3 brdfValue;
	if (!(wo.z*wh.z > 0.f))
	{
		wh = -wh;
	}
	float pdf = FresnelDielectric(absdot(wo, wh), state.ior);

	if (u2 < pdf)
	{
		wi = -wo + 2.f*abs(dot(wo, wh))*wh;
		brdfValue = state.diffuse*GGX_EvalBrdf(state, wo, wi);
	}
	else {
		pdf = 1.0f - pdf;
		bsdfState.bsdfEvent |= BsdfEvent_Refraction;
		float n = dot(wo, wh);
		float c = (1.0f / state.ior);
		wi = c * n - sign(dot(wo, state.frame.mZ))*sqrtf(1.f + n * (c*c - 1.f))*wh - n * wo;
		brdfValue = GGX_EvalBrdf_Refract(state, wo, wi)*state.diffuse;
	}

	val = state.frame.ToWorld(wi);

	bsdfState.pdfW += GGX_Pdf(wi, wo, state.frame.mZ, alpha);
	bsdfState.invPdf += GGX_Pdf(wo, wi, state.frame.mZ, alpha);

	return brdfValue;

}



RT_CALLABLE_PROGRAM float3 CT_Conductor_GGX_Microfacet_Eval(ShadingState const state, BsdfState& bsdfState, const float3 wo)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 lwo = state.frame.ToLocal(wo);

	bsdfState.pdfW += GGX_Pdf(lwo, state.IncomingDir, state.frame.mZ, GetAlpha(state));
	bsdfState.invPdf += GGX_Pdf(state.IncomingDir, lwo, state.frame.mZ, GetAlpha(state));


	return GGX_EvalBrdf_Conductor(state, state.IncomingDir, lwo);
}

RT_CALLABLE_PROGRAM float3 CT_Conductor_GGX_Microfacet_Sample(ShadingState const state, BsdfState& bsdfState, const float u0, const float u1, const float u2, float3& val)
{
	bsdfState.cosAt = abs(state.IncomingDir.z);
	bsdfState.bsdfEvent = BsdfEvent_Glossy;

	float3 wo = state.IncomingDir;
	float3 wi;

	float alpha = GetAlpha_GGX(state);
	float theta = atanf(sqrtf(alpha*alpha*logf(1.0f - u0)));
	float phi = u1 * 2.f*M_PIf;

	float3 wh = SphericalDir(sinf(theta), cosf(theta), phi);
	float3 brdfValue;
	if (!(wo.z*wh.z > 0.f))
	{
		wh = -wh;
	}

	wi = -wo + 2.f*abs(dot(wo, wh))*wh;
	brdfValue = GGX_EvalBrdf_Conductor(state, wo, wi);

	val = state.frame.ToWorld(wi);

	bsdfState.pdfW += GGX_Pdf(wi, wo, state.frame.mZ, alpha);
	bsdfState.invPdf += GGX_Pdf(wo, wi, state.frame.mZ, alpha);

	return brdfValue;

}