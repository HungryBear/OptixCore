using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeBuffer : OptixPrimeNode
    {
        /*
         *RTPbuffertype m_type;
  T* m_ptr;
  int m_device;
  size_t m_count;
  PageLockedState m_pageLockedState;

         *
         */
        private readonly PrimeBufferDesc _fmt;
        private IntPtr _dataPointer;
        private long size;

        public PrimeBuffer(PrimeContext context, PrimeBufferDesc fmt, IntPtr data) : base(context)
        {
            _fmt = fmt;
            _dataPointer = data;
            CheckError(PrimeApi.rtpBufferDescCreate(context.InternalPtr, (RTPbufferformat)fmt.Format, (RTPbuffertype)fmt.Type, data, out InternalPtr));
        }


        public static unsafe PrimeBuffer Create<T>(PrimeContext ctx, PrimeBufferDesc desc, T[] data)
            where T: struct
        {
            var dataToCopy = new Memory<T>(data);
            var data2Cpy = dataToCopy.Pin();
            var dataLength = Unsafe.SizeOf<T>() * data.Length;
            var dataPtr = IntPtr.Zero;
            if (desc.Type == RTPBufferType.CudaLinear)
            {
                //CudaDriverApi.CudaCall(CudaDriverApi.cuMemAlloc(ref dataPtr, (uint) dataLength));

                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cuMemAlloc(ref dataPtr, (uint)dataLength));
                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMemcpy(dataPtr.ToPointer(), new IntPtr(data2Cpy.Pointer), (uint) dataLength, CudaMemCpyKind.cudaMemcpyHostToDevice ));
                // CudaDriverApi.CudaCall(CudaDriverApi.cuMemcpyHtoD(ref dataPtr, data2Cpy.Pointer, (uint) dataLength));
            }
            else
            {
                dataPtr = Marshal.AllocHGlobal(dataLength);
                GC.AddMemoryPressure(dataLength);
                var span = new Span<T>(data);
                MemoryHelper.CopyFromManaged(ref span, dataPtr, (uint) data.Length);
               // Unsafe.Copy(dataPtr.ToPointer(), ref data);
            }
            data2Cpy.Dispose();
            return new PrimeBuffer(ctx, desc, dataPtr){size = dataLength};

        }

        public void SetRange(ulong begin, ulong end)
        {
            CheckError(PrimeApi.rtpBufferDescSetRange(ref InternalPtr, begin, end));
        }

        public void Lock()
        {
            if (_fmt.Type == RTPBufferType.CudaLinear)
                return;
            CheckError(PrimeApi.rtpHostBufferLock(InternalPtr, (ulong)size));
        }

        public void Unlock()
        {
            if (_fmt.Type == RTPBufferType.CudaLinear)
                return;
            CheckError(PrimeApi.rtpHostBufferUnlock(InternalPtr));
        }

        unsafe T[] GetDataInternal<T>()
        {
            var result = new T[size / Unsafe.SizeOf<T>()];
            
            if (_fmt.Type == RTPBufferType.Host)
            {
                var span = new Span<T>(_dataPointer.ToPointer(), (int) (size/Unsafe.SizeOf<T>()));
                return span.ToArray();
                //Unsafe.Copy(ref result, _dataPointer.ToPointer());
            }
            else
            {
                var memory = new Memory<T>(result);
                var mh = memory.Pin();
                //CudaDriverApi.CudaCall(CudaDriverApi.cuMemcpyDtoH(mh.Pointer, ref _dataPointer, (uint)size));

                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMemcpy(mh.Pointer, _dataPointer, (uint)size, CudaMemCpyKind.cudaMemcpyDeviceToHost));
                mh.Dispose();
            }

            return result;
        }

        public T[] GetData<T>()
        {
            return GetDataInternal<T>();
        }

        public override void Validate()
        {

        }

        public override void Destroy()
        {
            if (_fmt.Type == RTPBufferType.CudaLinear)
            {
                //CudaDriverApi.CudaCall(CudaDriverApi.cuMemFree(_dataPointer));
                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaFree(_dataPointer));

            }
            else
            {
                GC.RemoveMemoryPressure(size);
                Marshal.FreeHGlobal(_dataPointer);
            }

            CheckError(PrimeApi.rtpBufferDescDestroy(InternalPtr));
        }
    }
}