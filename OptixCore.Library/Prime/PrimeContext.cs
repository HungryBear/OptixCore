using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeContext : BasePrimeEntity, IDisposable
    {
        private List<IDisposable> _buffers;
        public PrimeContext(bool useCuda = false)
        {
            _buffers = new List<IDisposable>();
            CheckError(PrimeApi.rtpContextCreate(useCuda ? RTPcontexttype.RTP_CONTEXT_TYPE_CUDA : RTPcontexttype.RTP_CONTEXT_TYPE_CPU, out InternalPtr));
            CheckError(PrimeApi.rtpContextSetCudaDeviceNumbers(InternalPtr, 0, 0));
        }

        public string GetVersion()
        {
            CheckError(PrimeApi.rtpGetVersionString(out var str));
            return Marshal.PtrToStringAnsi(str);
        }

        public PrimeBuffer CreateBuffer<T>(RTPBufferType type, RtpBufferFormat format, T[] data)
            where T : struct
        {
            var desc = new PrimeBufferDesc { Type = type, Format = format };
            var buffer = PrimeBuffer.Create(this, desc, data);
            _buffers.Add(buffer);
            return buffer;
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
            foreach (var disposable in _buffers)
            {
                disposable.Dispose();
            }
            _buffers.Clear();
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(PrimeApi.rtpContextDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }

 
    }
}