using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeEngine : IDisposable
    {

        private PrimeContext _context;
        private PrimeModel _model;
        private PrimeQuery _query;

        private RayFormat _rayFormat;
        private RayHitType _hitType;
        private readonly RTPBufferType _bufferType;

        private PrimeBuffer _rayBuffer, _hitBuffer;

        public PrimeEngine(RayFormat rayFormat, RayHitType hitType, RTPBufferType bufferType=RTPBufferType.Host, bool useCPU = false)
        {
            _rayFormat = rayFormat;
            _hitType = hitType;
            _bufferType = bufferType;
            _context = new PrimeContext(!useCPU);
        }

        public void SetMesh(Vector3[] vertices, int[] indices)
        {
            if (_model == null)
            {
                _model = new PrimeModel(_context);
            }

            var vertexBuffer = _context.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.VERTEX_FLOAT3, vertices);
            var indexBuffer = _context.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.IndicesInt3, indices);
            indexBuffer.SetRange(0u, (uint)(indices.Length / 3));
            vertexBuffer.SetRange(0u, (uint)vertices.Length);

            _model.SetTriangles(indexBuffer, vertexBuffer);

            _model.Update(RTPmodelhint.RTP_MODEL_HINT_NONE);
            _model.Finish();
        }

        public void SetRays<T>(T[] rays)
            where T : struct
        {
            if (_query == null)
                _query = new PrimeQuery(_context, _model, QueryType.ClosestHit);

            if (_rayBuffer == null)
            {
                _rayBuffer = _context.CreateBuffer(_bufferType,
                    _rayFormat == RayFormat.OriginDirectionMinMaxInterleaved ?
                    RtpBufferFormat.RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX
                    : RtpBufferFormat.RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, rays);
                _rayBuffer.SetRange(0u, (ulong)rays.Length);
                _hitBuffer = _context.CreateBuffer(_bufferType,
                    RtpBufferFormat.RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, Enumerable.Repeat(new Hit { t = 1e-4f }, rays.Length).ToArray());
                _hitBuffer.SetRange(0u, (ulong)rays.Length);
                //_hitBuffer.Lock();
                _query.SetRays(_rayBuffer);
                _query.SetHits(_hitBuffer);
            }
            else
            {
                _rayBuffer.SetData(rays);
            }
        }

        public Hit[] Query()
        {
            _query.Execute(0);
            _query.Finish();

            return _hitBuffer.GetData<Hit>();
        }

        public void Dispose()
        {
            _query?.Dispose();
            _model?.Dispose();
            _context?.Dispose();
        }

        #region Internal Types
        [StructLayout(LayoutKind.Sequential)]
        public struct Ray
        {
            //internal const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

            public Vector3 origin;
            public float tmin;
            public Vector3 dir;
            public float tmax;
            public override string ToString()
            {
                return $"{origin}|{dir}|[{tmin},{tmax}]";
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct RayMinMax
        {
            public Vector3 origin;
            public Vector3 dir;

        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Hit
        {
            //internal const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

            public float t;
            public int triId;
            public float u;
            public float v;
            public override string ToString()
            {
                return $"T:{triId}|Dist:{t}|[{u},{v}]";
            }
        };

        public struct HitInstancing
        {
            // internal const RTPbufferformat format = RTPbufferformat.RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

            public float t;
            public int triId;
            public int instId;
            public float u;
            public float v;
        };
        #endregion

    }
}