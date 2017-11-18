using System;

namespace OptixCore.Library
{
    public struct Int2 : IEquatable<Int2>
    {
        public int X, Y;

        public Int2(int a)
        {
            X = Y = a;
        }

        public Int2(int x, int y)
        {
            X = x;
            Y = y;
        }

        public bool Equals(Int2 other)
        {
            return other.X.Equals(X) && other.Y.Equals(Y);
        }
    }
}