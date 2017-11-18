
using System.Numerics;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Utils
    {
        public static uint GetFormatSize(Format format)
        {
            Api.rtuGetSizeForRTformat((RTformat)format, out var size);
            return size;
        }


        public static Vector4 CreatePlane(Vector3 normal, Vector3 point)
        {
            normal = Vector3.Normalize(normal);
            float d = Vector3.Dot(-normal, point);

            return new Vector4(normal, d);
        }
    }
}