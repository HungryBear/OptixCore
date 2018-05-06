using System;
using System.Numerics;

namespace OptixCore.Library
{
    public struct BoundingBox
    {

        public Vector3 Min;
        public Vector3 Max;

        public BoundingBox(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }

        public void Invalidate()
        {
            Min = new Vector3(1e18f, 1e18f, 1e18f);
            Max = new Vector3(1e-10f, 1e-10f, 1e-10f);
        }

        public void AddFloat3(Vector3 p)
        {

            Min = new Vector3(System.Math.Min((Min.X), (p.X)),
                          System.Math.Min((Min.Y), (p.Y)),
                          System.Math.Min((Min.Z), (p.Z)));
            Max = new Vector3(System.Math.Max(Max.X, p.X),
                            System.Math.Max(Max.Y, p.Y),
                            System.Math.Max(Max.Z, p.Z));
        }

        public bool Contains(Vector3 p)
        {
            return p.X >= Min.X && p.X <= Max.X &&
                    p.Y >= Min.Y && p.Y <= Max.Y &&
                    p.Z >= Min.Z && p.Z <= Max.Z;
        }

        public void TranslateSelf(Vector3 trans)
        {
            Min += trans;
            Max += trans;
        }

        public Vector3 Extent()
        {
            return Max - Min;
        }

        public float Extent(int dim)
        {
            if (dim == 0)
            {
                return Max.X - Min.X;
            }
            if (dim == 1)
            {
                return Max.Y - Min.Y;
            }

            if (dim == 2)
            {
                return Max.Z - Min.Z;
            }
            throw new ArgumentException(nameof(dim));
        }

        public int LongestAxis()
        {
            Vector3 d = Extent();

            if (d.X > d.Y)
                return d.X > d.Z ? 0 : 2;

            return d.Y > d.Z ? 1 : 2;
        }

        public float MaxExtent()
        {
            return Extent(LongestAxis());
        }

        public Vector3 Center => (Min + Max) * 0.5f;

        public bool IsValid => Min.X <= Max.X &&
                               Min.Y <= Max.Y &&
                               Min.Z <= Max.Z;

        public static int SizeInBytes => System.Runtime.InteropServices.Marshal.SizeOf<BoundingBox>();


        public static BoundingBox Invalid
        {
            get
            {
                var box = new BoundingBox();
                box.Invalidate();

                return box;
            }
        }

        public Vector3 this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return Min;
                    case 1:
                        return Max;
                    default:
                        throw new ArgumentOutOfRangeException("index", "index outside the number of bounding box components");
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        Min = value;
                        break;
                    case 1:
                        Max = value;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException("index", "Index outside the number of bounding box components");
                }
            }
        }
    }
}