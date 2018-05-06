using System;
using System.Runtime.CompilerServices;
using OptixCore.Library.Native;

namespace OptixCore.Library.Scene
{
    using Buffer = OptixBuffer;
    public class FilteringManager
    {

        #region Entities

        public abstract class Filter
        {
            public Filter(float xw, float yw)
            {
                xWidth = (xw);
                yWidth = (yw);
                invXWidth = (1f / xw);
                invYWidth = (1f / yw);
            }


            public abstract float Evaluate(float x, float y);

            // Filter Public Data
            public float xWidth, yWidth;
            public float invXWidth, invYWidth;
        };

        public class GaussianFilter : Filter
        {
            public GaussianFilter(float xw = 2f, float yw = 2f, float a = 2f)
                : base(xw, yw)
            {
                alpha = a;
                expX = (float)Math.Exp(-alpha * xWidth * xWidth);
                expY = (float)Math.Exp(-alpha * yWidth * yWidth);
            }


            public override sealed float Evaluate(float x, float y)
            {
                return Gaussian(x, expX) * Gaussian(y, expY);
            }


            float alpha;
            float expX, expY;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            float Gaussian(float d, float expv)
            {
                return Math.Max(0f, (float)Math.Exp(-alpha * d * d) - expv);
            }
        };

        public class BoxFilter : Filter
        {
            public BoxFilter(float xw, float yw)
                : base(xw, yw)
            {
            }

            public override sealed float Evaluate(float x, float y)
            {
                return 1f;
            }
        }


        public sealed class MitchelFilter : Filter
        {
            public MitchelFilter(float xw, float yw, float b = 1f / 3f, float c = 1f / 3f)
                : base(xw, yw)
            {
                this.B = b;
                this.C = c;
            }

            public override float Evaluate(float x, float y)
            {
                var distance = (float)Math.Sqrt(x * x * invXWidth * invXWidth + y * y * invYWidth * invYWidth);

                return Mitchell1D(distance);

            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            float Mitchell1D(float x)
            {
                if (x >= 1f)
                    return 0f;
                x = Math.Abs(2f * x);
                if (x > 1f)
                    return (((-B / 6f - C) * x + (B + 5f * C)) * x +
                        (-2f * B - 8f * C)) * x + (4f / 3f * B + 4f * C);
                else
                    return ((2f - 1.5f * B - C) * x +
                        (-3f + 2f * B + C)) * x * x +
                        (1f - B / 3f);
            }

            float B, C;
        }

        public class FilterLUT
        {
            public FilterLUT(Filter filter, float offsetX, float offsetY)
            {
                int x0 = Ceil2UInt(offsetX - filter.xWidth);
                int x1 = Floor2UInt(offsetX + filter.xWidth);
                int y0 = Ceil2UInt(offsetY - filter.yWidth);
                int y1 = Floor2UInt(offsetY + filter.yWidth);
                lutWidth = x1 - x0 + 1;
                lutHeight = y1 - y0 + 1;
                lut = new float[lutWidth * lutHeight];

                float filterNorm = 0f;
                int index = 0;
                for (int iy = y0; iy <= y1; ++iy)
                {
                    for (int ix = x0; ix <= x1; ++ix)
                    {
                        float filterVal = filter.Evaluate(Math.Abs(ix - offsetX), Math.Abs(iy - offsetY));
                        filterNorm += filterVal;
                        lut[index++] = filterVal;
                    }
                }

                // Normalize LUT
                filterNorm = 1f / filterNorm;
                index = 0;
                for (int iy = y0; iy <= y1; ++iy)
                {
                    for (int ix = x0; ix <= x1; ++ix)
                        lut[index++] *= filterNorm;
                }
            }


            public int GetWidth()
            {
                return lutWidth;
            }
            public int GetHeight()
            {
                return lutHeight;
            }

            public float[] GetLUT()
            {
                return lut;
            }


            int lutWidth, lutHeight;
            float[] lut;
        };


        public class FilterLUTs
        {
            public FilterLUTs(Filter filter, int size)
            {
                lutsSize = size + 1;
                step = 1f / (float)(size);

                luts = new FilterLUT[lutsSize * lutsSize];

                for (var iy = 0; iy < lutsSize; ++iy)
                {
                    for (var ix = 0; ix < lutsSize; ++ix)
                    {
                        var x = ix * step - 0.5f + step / 2f;
                        var y = iy * step - 0.5f + step / 2f;
                        luts[ix + iy * lutsSize] = new FilterLUT(filter, x, y);
                    }
                }
            }


            public FilterLUT GetLUT(float x, float y)
            {
                int ix = Math.Max(0, Math.Min(Floor2UInt(lutsSize * (x + 0.5f)), lutsSize - 1));
                int iy = Math.Max(0, Math.Min(Floor2UInt(lutsSize * (y + 0.5f)), lutsSize - 1));
                return luts[ix + iy * lutsSize];
            }

            private
                int lutsSize;
            float step;
            FilterLUT[] luts;
        };
        #endregion


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Floor2UInt(float val)
        {
            if ((double)val <= 0.0)
                return 0;
            return (int)val;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Ceil2UInt(float val)
        {
            if ((double)val <= 0.0)
                return 0;
            else
                return (int)Math.Ceiling((double)val);
        }
        public static Buffer CreateLutBuffer(Context optixContext, float f_x, float f_y)
        {
            var filter =
                new MitchelFilter(f_x, f_y);
            //new GaussianFilter(f_x, f_y);

            float[] Gaussian2x2_filterTable = new float[16 * 16];
            for (var y = 0; y < 16; ++y)
            {
                float fy = (y + 0.5f) * 2.0f / 16.0f;
                for (var x = 0; x < 16; ++x)
                {
                    float fx = (x + 0.5f) * 2.0f / 16.0f;
                    Gaussian2x2_filterTable[x + y * 16] = filter.Evaluate(fx, fy);
                }
            }

            BufferDesc desc = new BufferDesc
            {
                Width = 16 * 16,
                Type = BufferType.Input,
                Format = Format.Float
            };

            Buffer res = new Buffer(optixContext, desc);

            res.SetData(Gaussian2x2_filterTable);

            return res;
        }
    }
}