using System;
using System.Runtime.InteropServices;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeBuffer : OptixPrimeNode
    {
        private IntPtr _dataPointer;
        public PrimeBuffer(PrimeContext context, PrimeBufferDesc fmt, IntPtr data) : base(context)
        {
            _dataPointer = data;
            CheckError(PrimeApi.rtpBufferDescCreate(context.InternalPtr, (RTPbufferformat)fmt.Format, (RTPbuffertype)fmt.Type, data, out InternalPtr ));            
        }

        public static PrimeBuffer Create<T>(PrimeContext ctx, PrimeBufferDesc desc, T[] data)
        {
            var h = GCHandle.Alloc(data, GCHandleType.Pinned);
            var span = new Span<T>(data);
            //MemoryHelper.CopyFromManaged(ref span, );
            throw new NotImplementedException();
        }

        public override void Validate()
        {
            
        }

        public override void Destroy()
        {
            CheckError(PrimeApi.rtpBufferDescDestroy(InternalPtr));
        }
    }
}