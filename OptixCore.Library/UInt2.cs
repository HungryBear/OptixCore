using System;

namespace OptixCore.Library
{
    public struct UInt2 : IEquatable<UInt2>
    {
        public uint X, Y;

        public UInt2(uint a)
        {
            X = Y = a;
        }

        public UInt2(uint x, uint y)
        {
            X = x;
            Y = y;
        }

        public bool Equals(UInt2 other)
        {
            return other.X.Equals(X) && other.Y.Equals(Y);
        }
    }
}