using System;
using System.Runtime.InteropServices;

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

    public struct UByte4 : IEquatable<UByte4>
    {
        public byte X;
        public byte Y;
        public byte Z;
        public byte W;

        public UByte4(byte x, byte y, byte z, byte w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        public static int SizeInBytes => Marshal.SizeOf<UByte4>();

        public override string ToString()
        {
            return $"{{ {X} {Y} {Z} {W} }}";
        }

        public bool Equals(UByte4 other)
        {
            return X == other.X && Y == other.Y && Z == other.Z && W == other.W;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is UByte4 byte4 && Equals(byte4);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = X.GetHashCode();
                hashCode = (hashCode * 397) ^ Y.GetHashCode();
                hashCode = (hashCode * 397) ^ Z.GetHashCode();
                hashCode = (hashCode * 397) ^ W.GetHashCode();
                return hashCode;
            }
        }
    }
}