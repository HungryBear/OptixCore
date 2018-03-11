#pragma once

#include <optix_math.h>

template<typename T>
static __device__ __forceinline__ int sign(T val)
{
	return (T(0) < val) - (val < T(0));
}

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __CUDACC__
__device__ __inline__ uchar4 make_color(const float3& c)
{
	return make_uchar4(static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
		static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
		static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
		255u);                                                 /* A */
}

__device__ __inline__ uchar4 make_color(const float4& c)
{
	return make_uchar4(static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
		static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
		static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
		static_cast<unsigned char>(__saturatef(c.w)*255.99f)); /* A */
}

__device__ __inline__ uchar4 bgra_to_rgba(const uchar4& c)
{
	return make_uchar4(c.x, c.y, c.z, c.w);
}

__device__ __inline__ uchar4 rgba_to_bgra(const uchar4& c)
{
	return make_uchar4(c.z, c.y, c.x, c.w);
}
#endif

__device__ __inline__ float radians(float f)
{
	return (f*M_PIf / 180.0f);
}

// Calculate the luminance value of an rgb triple
__device__ __inline__ float luminance(const float3& rgb)
{
	const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
	return  dot(rgb, ntsc_luminance);
}

// Maps concentric squares to concentric circles (Shirley and Chiu)
__host__ __device__ __inline__ float2 square_to_disk(float2 sample)
{
	float phi, r;

	const float a = 2.0f * sample.x - 1.0f;
	const float b = 2.0f * sample.y - 1.0f;

	if (a > -b)
	{
		if (a > b)
		{
			r = a;
			phi = (float)M_PI_4f * (b / a);
		}
		else
		{
			r = b;
			phi = (float)M_PI_4f * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (float)M_PI_4f * (4.0f + (b / a));
		}
		else
		{
			r = -b;
			phi = (b) ? (float)M_PI_4f * (6.0f - (a / b)) : 0.0f;
		}
	}

	return make_float2(r * cosf(phi), r * sinf(phi));
}

// Convert cartesian coordinates to polar coordinates
__host__ __device__ __inline__ float3 cart_to_pol(float3 v)
{
	float azimuth;
	float elevation;
	float radius = length(v);

	float r = sqrtf(v.x*v.x + v.y*v.y);
	if (r > 0.0f)
	{
		azimuth = atanf(v.y / v.x);
		elevation = atanf(v.z / r);

		if (v.x < 0.0f)
			azimuth += M_PIf;
		else if (v.y < 0.0f)
			azimuth += M_PIf * 2.0f;
	}
	else
	{
		azimuth = 0.0f;

		if (v.z > 0.0f)
			elevation = +M_PI_2f;
		else
			elevation = -M_PI_2f;
	}

	return make_float3(azimuth, elevation, radius);
}

// Sample Phong lobe relative to U, V, W frame
__host__ __device__ __inline__ float3 sample_phong_lobe(float2 sample, float exponent, float3 U, float3 V, float3 W)
{
	const float power = expf(logf(sample.y) / (exponent + 1.0f));
	const float phi = sample.x * 2.0f * (float)M_PIf;
	const float scale = sqrtf(1.0f - power * power);

	const float x = cosf(phi)*scale;
	const float y = sinf(phi)*scale;
	const float z = power;

	return x * U + y * V + z * W;
}

static
__host__ __device__ __inline__ optix::float3 sample_phong_lobe(const optix::float2 &sample, float exponent,
	const optix::float3 &U, const optix::float3 &V, const optix::float3 &W,
	float &pdf, float &bdf_val)
{
	const float cos_theta = powf(sample.y, 1.0f / (exponent + 1.0f));

	const float phi = sample.x * 2.0f * M_PIf;
	const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	const float x = cosf(phi)*sin_theta;
	const float y = sinf(phi)*sin_theta;
	const float z = cos_theta;

	const float powered_cos = powf(cos_theta, exponent);
	pdf = (exponent + 1.0f) / (2.0f*M_PIf) * powered_cos;
	bdf_val = (exponent + 2.0f) / (2.0f*M_PIf) * powered_cos;

	return x * U + y * V + z * W;
}

// Create ONB from normalalized vector
__device__ __inline__ void createONB(const optix::float3& n,
	optix::float3& U,
	optix::float3& V)
{
	using namespace optix;

	U = cross(n, make_float3(0.0f, 1.0f, 0.0f));
	if (dot(U, U) < 1.e-3f)
		U = cross(n, make_float3(1.0f, 0.0f, 0.0f));
	U = normalize(U);
	V = cross(n, U);
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void create_onb(const float3& n, float3& U, float3& V, float3& W)
{
	W = normalize(n);
	U = cross(W, make_float3(0.0f, 1.0f, 0.0f));

	if (fabs(U.x) < 0.001f && fabs(U.y) < 0.001f && fabs(U.z) < 0.001f)
		U = cross(W, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(W, U);
}

// Create ONB from normalized vector
__device__ __inline__ void create_onb(const float3& n, float3& U, float3& V)
{
	U = cross(n, make_float3(0.0f, 1.0f, 0.0f));

	if (dot(U, U) < 1e-3f)
		U = cross(n, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(n, U);
}

// Compute the origin ray differential for transfer
__host__ __device__ __inline__ float3 differential_transfer_origin(float3 dPdx, float3 dDdx, float t, float3 direction, float3 normal)
{
	float dtdx = -dot((dPdx + t * dDdx), normal) / dot(direction, normal);
	return (dPdx + t * dDdx) + dtdx * direction;
}

// Compute the direction ray differential for a pinhole camera
__host__ __device__ __inline__ float3 differential_generation_direction(float3 d, float3 basis)
{
	float dd = dot(d, d);
	return (dd*basis - dot(d, basis)*d) / (dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
__host__ __device__ __inline__ float3 differential_reflect_direction(float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N)
{
	float3 dNdx = dNdP * dPdx;
	float dDNdx = dot(dDdx, N) + dot(D, dNdx);
	return dDdx - 2 * (dot(D, N)*dNdx + dDNdx * N);
}

// Compute the direction ray differential for refraction
__host__ __device__ __inline__ float3 differential_refract_direction(float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N, float ior, float3 T)
{
	float eta;
	if (dot(D, N) > 0.f) {
		eta = ior;
		N = -N;
	}
	else {
		eta = 1.f / ior;
	}

	float3 dNdx = dNdP * dPdx;
	float mu = eta * dot(D, N) - dot(T, N);
	float TN = -sqrtf(1 - eta * eta*(1 - dot(D, N)*dot(D, N)));
	float dDNdx = dot(dDdx, N) + dot(D, dNdx);
	float dmudx = (eta - (eta*eta*dot(D, N)) / TN)*dDNdx;
	return eta * dDdx - (mu*dNdx + dmudx * N);
}

template <class T>
__host__ __device__ __inline__ float3 bilerp(float u, float v,
	const T& x00, const T& x10,
	const T& x01, const T& x11)
{
	return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

__host__ __device__ __inline__ float3 Yxy2XYZ(const float3& Yxy)
{
	return make_float3(Yxy.y * (Yxy.x / Yxy.z),
		Yxy.x,
		(1.0f - Yxy.y - Yxy.z) * (Yxy.x / Yxy.z));
}



__host__ __device__ __inline__ float3 Yxy2rgb(float3 Yxy)
{
	// First convert to xyz
	float3 xyz = make_float3(Yxy.y * (Yxy.x / Yxy.z),
		Yxy.x,
		(1.0f - Yxy.y - Yxy.z) * (Yxy.x / Yxy.z));

	const float R = dot(xyz, make_float3(3.2410f, -1.5374f, -0.4986f));
	const float G = dot(xyz, make_float3(-0.9692f, 1.8760f, 0.0416f));
	const float B = dot(xyz, make_float3(0.0556f, -0.2040f, 1.0570f));
	return make_float3(R, G, B);
}


__host__ __device__ __inline__ float3 rgb2Yxy(float3 rgb)
{
	// convert to xyz
	const float X = dot(rgb, make_float3(0.4124f, 0.3576f, 0.1805f));
	const float Y = dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
	const float Z = dot(rgb, make_float3(0.0193f, 0.1192f, 0.9505f));

	// convert xyz to Yxy
	return make_float3(Y,
		X / (X + Y + Z),
		Y / (X + Y + Z));
}

__device__ __inline__ float3 RgbToXyz(const float3& p) {
	float3 rgb = p;
	if (rgb.x <= 0.04045f)
	{
		rgb.x /= 12.92f;
	}
	else {
		rgb.x = powf((rgb.x + 0.055f) / 1.055f, 2.2f);
	}

	if (rgb.y <= 0.04045f)
	{
		rgb.y /= 12.92f;
	}
	else {
		rgb.y = powf((rgb.y + 0.055f) / 1.055f, 2.2f);
	}

	if (rgb.z <= 0.04045f)
	{
		rgb.z /= 12.92f;
	}
	else {
		rgb.z = powf((rgb.z + 0.055f) / 1.055f, 2.2f);
	}
	float3 xyz;
	xyz.x = 0.412453f*rgb.x + 0.357580f*rgb.y + 0.180423f*rgb.z;
	xyz.y = 0.212671f*rgb.x + 0.715160f*rgb.y + 0.072169f*rgb.z;
	xyz.z = 0.019334f*rgb.x + 0.119193f*rgb.y + 0.950227f*rgb.z;
	return xyz;
}

__host__ __device__ __inline__ float3 XyzToRgb(const float3& xyz)
{
	float R = dot(xyz, make_float3(3.2410f, -1.5374f, -0.4986f));
	float G = dot(xyz, make_float3(-0.9692f, 1.8760f, 0.0416f));
	float B = dot(xyz, make_float3(0.0556f, -0.2040f, 1.0570f));

	if (R <= 0.0031308f) {
		R *= 12.92f;
	}
	else {
		R = powf(1.055f*R, 1.0f / 2.2f) - 0.055f;
	}

	if (G <= 0.0031308f) {
		G *= 12.92f;
	}
	else {
		G = powf(1.055f*G, 1.0f / 2.2f) - 0.055f;
	}

	if (B <= 0.0031308f) {
		B *= 12.92f;
	}
	else {
		B = powf(1.055f*B, 1.0f / 2.2f) - 0.055f;
	}
	return make_float3(R, G, B);
}


__device__ __inline__ float3 RGB2XYZ(const float3& rgb) {
	float3 xyz;
	xyz.x = 0.412453f*rgb.x + 0.357580f*rgb.y + 0.180423f*rgb.z;
	xyz.y = 0.212671f*rgb.x + 0.715160f*rgb.y + 0.072169f*rgb.z;
	xyz.z = 0.019334f*rgb.x + 0.119193f*rgb.y + 0.950227f*rgb.z;
	return xyz;
}

__host__ __device__ __inline__ float3 XYZ2Yxy(const float3& xyz)
{
	return make_float3(xyz.y, xyz.x / (xyz.x + xyz.y + xyz.z), xyz.y / (xyz.x + xyz.y + xyz.z));
}

__host__ __device__ __inline__ float3 XYZ2rgb(const float3& xyz)
{
	const float R = dot(xyz, make_float3(3.2410f, -1.5374f, -0.4986f));
	const float G = dot(xyz, make_float3(-0.9692f, 1.8760f, 0.0416f));
	const float B = dot(xyz, make_float3(0.0556f, -0.2040f, 1.0570f));
	return make_float3(R, G, B);
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void createONB(const float3& n, float3& U, float3& V, float3& W)
{
	W = normalize(n);
	U = cross(W, make_float3(0.0f, 1.0f, 0.0f));

	if (abs(U.x) < 0.001f && abs(U.y) < 0.001f && abs(U.z) < 0.001f)
		U = cross(W, make_float3(1.0f, 0.0f, 0.0f));

	U = normalize(U);
	V = cross(W, U);
}

// sample hemisphere with cosine density
__device__ __inline__ float3 sampleUnitHemisphere(const float2& sample,
	const float3& U,
	const float3& V,
	const float3& W)
{
	float phi = 2.0f * M_PIf * sample.x;
	float r = sqrt(sample.y);
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrtf(max(0.0f, 1.0f - x * x - y * y));

	return x * U + y * V + z * W;
}

__device__ __inline__ float3 pow3f(float3 x, float y)
{
	x.x = powf(x.x, y);
	x.y = powf(x.y, y);
	x.z = powf(x.z, y);

	return x;
}


// sample hemisphere with cosine density


__device__ __inline__ float3 sampleUnitSphere(const float2& sample)
{
	float z = 1.0f - 2.0f * sample.x;
	float r = sqrtf(max(0.0f, 1.0f - z * z));
	float phi = 2.0f * M_PIf * sample.y;

	float x = cos(phi);
	float y = sin(phi);

	return make_float3(x, y, z);
}

__device__ __inline__ float3 sampleSphere(const float2& sample,
	const float3& U,
	const float3& V,
	const float3& W)
{
	float z = 1.0f - 2.0f * sample.x;
	float r = sqrtf(max(0.0f, 1.0f - z * z));
	float phi = 2.0f * M_PIf * sample.y;

	float x = cos(phi);
	float y = sin(phi);

	return x * U + y * V + z * W;
}

//Reinhard's tone mapping
static __host__ __device__ __inline__ optix::float3 tonemap(const optix::float3 &hdr_value, float Y_log_av, float Y_max, float a = 0.04)
{
	using namespace optix;

	float3 val_Yxy = rgb2Yxy(hdr_value);

	float Y = val_Yxy.x; // Y channel is luminance
	float Y_rel = a * Y / Y_log_av;
	float mapped_Y = Y_rel * (1.0f + Y_rel / (Y_max * Y_max)) / (1.0f + Y_rel);

	float3 mapped_Yxy = make_float3(mapped_Y, val_Yxy.y, val_Yxy.z);
	float3 mapped_rgb = Yxy2rgb(mapped_Yxy);

	return mapped_rgb;
}

static __host__ __device__ __inline__ optix::float3 tonemap_xyz(const optix::float3 &hdr_value, float Y_log_av, float Y_max, float a = 0.04)
{
	using namespace optix;

	float3 val_Yxy = XYZ2Yxy(hdr_value);

	float Y = val_Yxy.x; // Y channel is luminance
	float Y_rel = a * Y / Y_log_av;
	float mapped_Y = Y_rel * (1.0f + Y_rel / (Y_max * Y_max)) / (1.0f + Y_rel);

	float3 mapped_Yxy = make_float3(mapped_Y, val_Yxy.y, val_Yxy.z);
	float3 mapped_rgb = Yxy2rgb(mapped_Yxy);

	return mapped_rgb;
}


__device__ __inline__ float3 norm_rgb(const float4& c)
{
	return make_float3(c.z, c.y, c.x);
}

__device__ inline float3 powf(float3 a, float exp)
{
	return make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
}


__device__ inline float3 exp3f(float3 a, float exp)
{
	return make_float3(expf(a.x*exp), expf(a.y* exp), expf(a.z*exp));
}
__device__ inline float4 powf(float4 a, float exp)
{
	return make_float4(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp), powf(a.w, exp));
}


__device__ __inline__ float4 gamma(float4 cx, float gamma_value = 2.0f)
{
	float4 c = make_float4(powf(make_float3(cx), 1.0f / gamma_value), 0.0f);
	return make_float4(__saturatef(c.x), __saturatef(c.y), __saturatef(c.z), 0.f);
}

__device__ __inline__ float lRgbTosRgb(float c, float gamma_value = 2.0f)
{
	if (c <= 0.0031308f)
	{
		return 12.92f*c;
	}
	const float a = 0.055f;
	return (1.f + a)*powf(c, 1.0f / gamma_value) - a;
}
__device__ __inline__ float4 Gamma(float4 c, float gamma_value = 2.0f)
{
	return make_float4(__saturatef(lRgbTosRgb(c.x, gamma_value)), __saturatef(lRgbTosRgb(c.y, gamma_value)), __saturatef(lRgbTosRgb(c.z, gamma_value)), lRgbTosRgb(c.w, gamma_value));
}

static __device__ __inline__ float fold(const float value)
{
	return fminf(value, 1.0f - value) * 2.0f;
}


__device__ __inline__  float sphere_phi(const float3& w)
{
	return M_PIf + atan2(w.y, w.x);
}

__device__ __inline__  float sphere_theta(const float3& w)
{
	return acosf(w.z);
}

__device__ __inline__  float cos_theta(const float3 &w)
{
	return w.z;
};
__device__ __inline__  float abs_cos_theta(const float3 &w)
{
	return fabsf(w.z);
};
__device__ __inline__  float sin_theta2(const float3 &w)
{
	return max(0.f, 1.f - cos_theta(w)*cos_theta(w));
}


__device__ __inline__  float sin_theta(const float3 &w) {
	return sqrtf(sin_theta2(w));
}


__device__ __inline__  float cos_phi(const float3 &w) {
	float sintheta = sin_theta(w);
	if (sintheta == 0.f) return 1.f;
	return clamp(w.x / sintheta, -1.f, 1.f);
}

__device__ __inline__  float sin_phi(const float3 &w) {
	float sintheta = sin_theta(w);
	if (sintheta == 0.f) return 0.f;
	return clamp(w.y / sintheta, -1.f, 1.f);
}

__device__ __inline__ float geomFactor(const float3& p1, const float3& n1, const float3& p2, const float3& n2)
{

	float3 w = normalize(p2 - p1);
	return (abs(dot(n1, w))*abs(dot(n2, -w))) / (length(p2 - p1)*length(p2 - p1));
}

__device__ __inline__ float __geomFactor(const float3& p1, const float3& n1, const float3& p2, const float3& n2)
{
	const float3 l = p2 - p1;
	const float3 w = normalize(l);
	const float ndl = dot(n1, w);
	if (ndl <= 0.0f)
	{
		return 0.0f;
	}
	const float nldl = dot(n2, -w);
	if (nldl <= 0.0f)
	{
		return 0.0f;
	}
	return (ndl*nldl) / (length(l)*length(l));
}

__device__ __inline__ bool isNan(const float3& v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

__device__ __inline__ bool isFinite(const float3& v)
{
	return isfinite(v.x) || isfinite(v.y) || isfinite(v.z);
}

__device__ __inline__ float FrDiel(float cosi, float cost, const float etai, const float etat)
{
	float Rparl = ((etat * cosi) - (etai * cost)) /
		((etat * cosi) + (etai * cost));
	float Rperp = ((etai * cosi) - (etat * cost)) /
		((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp * Rperp) / 2.f;
}


__device__ __inline__ float FrCond(float cosi, const float &eta, const float &k)
{
	float tmp = (eta*eta + k * k) * cosi*cosi;
	float Rparl2 = (tmp - (2.f * eta * cosi) + 1) /
		(tmp + (2.f * eta * cosi) + 1);
	float tmp_f = eta * eta + k * k;
	float Rperp2 =
		(tmp_f - (2.f * eta * cosi) + cosi * cosi) /
		(tmp_f + (2.f * eta * cosi) + cosi * cosi);
	return (Rparl2 + Rperp2) / 2.f;
}

__device__ __forceinline__ float absdot(const float3& a, const float3& b)
{
	return abs(dot(a, b));
}

__device__ __forceinline__ float3 abs(const float3& a)
{
	return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

__device__ __forceinline__  bool intersect_sphere(const optix::Ray ray, const float4 sphere, float& tmin, float& tmax, float3& shading_normal)
{
	float3 center = make_float3(sphere);
	float3 O = ray.origin - center;
	float3 D = ray.direction;
	float radius = sphere.w;

	float b = dot(O, D);
	float c = dot(O, O) - radius * radius;
	float disc = b * b - c;
	if (disc > 0.0f) {
		float sdisc = sqrtf(disc);
		float root1 = (-b - sdisc);
		float root11 = 0.0f;
		// refine root1
		float3 O1 = O + root1 * ray.direction;
		b = dot(O1, D);
		c = dot(O1, O1) - radius * radius;
		disc = b * b - c;

		if (disc > 0.0f) {
			sdisc = sqrtf(disc);
			root11 = (-b - sdisc);
		}
		else
			return false;
		float root2 = (-b + sdisc) + root1;

		tmin = root1 + root11;
		tmax = root2;
		if (tmin > tmax)
		{
			tmax = root1 + root11;
			tmin = root2;
		}
		if (tmin < 0)
		{
			tmin = tmax;
			if (tmin < 0)
				return false;
		}
		shading_normal = (O + tmin * D) / radius;
		return true;
	}
	return false;
}