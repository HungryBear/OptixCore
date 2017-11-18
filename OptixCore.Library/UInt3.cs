using System;

namespace OptixCore.Library
{
    public struct UInt3 : IEquatable<UInt3>
    {
        public uint X, Y, Z;

        public UInt3(uint a)
        {
            X = Y = Z = a;
        }

        public UInt3(uint x, uint y, uint z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public bool Equals(UInt3 other)
        {
            return other.X.Equals(X) && other.Y.Equals(Y) && other.Z.Equals(Z);
        }
    }
}