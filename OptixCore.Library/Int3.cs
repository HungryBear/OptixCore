using System;

namespace OptixCore.Library
{
    public struct Int3 : IEquatable<Int3>
    {
        public int X, Y, Z;

        public Int3(int a)
        {
            X = Y = Z = a;            
        }

        public Int3(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public bool Equals(Int3 other)
        {
            return other.X.Equals(X) && other.Y.Equals(Y) && other.Z.Equals(Z);
        }
    }
}