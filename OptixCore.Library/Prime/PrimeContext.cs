using System;
using OptixCore.Library.Native;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeContext : IDisposable
    {
        protected internal IntPtr InternalPtr;

        public PrimeContext()
        {
            CheckError(PrimeApi.rtpContextCreate(RTPcontexttype.RTP_CONTEXT_TYPE_CUDA, out InternalPtr));
        }

        public PrimeBuffer CreateBuffer<T>(RTPBufferType type, RtpBufferFormat format, T[] data)
            where T: struct
        {
            var desc = new PrimeBufferDesc { Type = type, Format = format};
            return PrimeBuffer.Create(this, desc, data);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void InitCuda()
        {
            //CudaDriverApi.CudaCall(CudaDriverApi.cuInit(0));
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Destroy();
            }
        }

        private void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(PrimeApi.rtpContextDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }

        internal void CheckError(RTPresult result)
        {
            if (result != RTPresult.RTP_SUCCESS)
            {
                PrimeApi.rtpGetErrorString(result, out var message);
                throw new OptixException($"Optix context error : {message}");
            }
        }
    }
}