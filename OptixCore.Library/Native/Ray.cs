using System.Numerics;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Native
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Ray
    {
        public Vector3 origin;
        public Vector3 direction;
        public float tmin;
        public float tmax;
        public int ray_type;
    }
}