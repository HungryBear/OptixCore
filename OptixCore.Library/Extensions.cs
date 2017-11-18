using System;
using System.Numerics;

namespace OptixCore.Library
{
    public static class Extensions
    {
        public static float Get(this Vector3 c, int dim)
        {
            if (dim == 0)
                return c.X;
            if (dim == 1)
                return c.Y;
            if (dim == 2)
                return c.Z;
            throw new ArgumentException(nameof(dim));
        }

        public static void Set(this Vector3 c, int dim, float v)
        {
            if (dim == 0)
                c.X = v;
            if (dim == 1)
                c.Y = v;
            if (dim == 2)
                c.Z = v;
            throw new ArgumentException(nameof(dim));
        }

        public static void Add(this Vector3 c, int dim, float v)
        {
            if (dim == 0)
                c.X += v;
            if (dim == 1)
                c.Y += v;
            if (dim == 2)
                c.Z += v;
            throw new ArgumentException(nameof(dim));
        }
    }
}