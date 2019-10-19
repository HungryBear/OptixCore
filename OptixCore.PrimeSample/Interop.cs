using System.Numerics;
using OptixCore.Library.Native.Prime;

namespace OptixCore.PrimeSample
{
    struct Ray
    {
        public const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

        public Vector3 origin;
        public float tmin;
        public Vector3 dir;
        public float tmax;
    };

    struct Hit
    {
        public const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

        public float t;
        public int triId;
        public float u;
        public float v;
    };

    struct HitInstancing
    {
        const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

        public float t;
        public int triId;
        public int instId;
        public float u;
        public float v;
    };


    
}
