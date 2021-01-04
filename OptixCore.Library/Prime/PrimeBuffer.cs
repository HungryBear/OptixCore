using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeBuffer : OptixPrimeNode
    {

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
            where T : struct
        {
            var dataToCopy = new Memory<T>(data);
            var data2Cpy = dataToCopy.Pin();
            var dataLength = Unsafe.SizeOf<T>() * data.Length;
            var dataPtr = IntPtr.Zero;
            if (desc.Type == RTPBufferType.CudaLinear)
            {
                //CudaDriverApi.CudaCall(CudaDriverApi.cuMemAlloc(ref dataPtr, (uint) dataLength));

                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMalloc(ref dataPtr, (uint)dataLength));
                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMemcpy(dataPtr.ToPointer(), new IntPtr(data2Cpy.Pointer), (uint)dataLength, CudaMemCpyKind.cudaMemcpyHostToDevice));
                // CudaDriverApi.CudaCall(CudaDriverApi.cuMemcpyHtoD(ref dataPtr, data2Cpy.Pointer, (uint) dataLength));
            }
            else
            {
                dataPtr = Marshal.AllocHGlobal(dataLength);
                GC.AddMemoryPressure(dataLength);
                var span = new Span<T>(data);
                MemoryHelper.CopyFromManaged(ref span, dataPtr, (uint)data.Length);
                // Unsafe.Copy(dataPtr.ToPointer(), ref data);
            }
            data2Cpy.Dispose();
            
            var buffer =  new PrimeBuffer(ctx, desc, dataPtr) { size = dataLength };
            return buffer;

        }

        public void SetRange(ulong begin, ulong end)
        {
            CheckError(PrimeApi.rtpBufferDescSetRange(InternalPtr, begin, end));
        }

        public void SetStride(uint bytes)
        {
            CheckError(PrimeApi.rtpBufferDescSetStride(InternalPtr, bytes));
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

            if (_fmt.Type == RTPBufferType.Host)
            {
                var span = new Span<T>(_dataPointer.ToPointer(), (int)(size / Unsafe.SizeOf<T>()));
                return span.ToArray();
                //Unsafe.Copy(ref result, _dataPointer.ToPointer());
            }
            var result = new T[size / Unsafe.SizeOf<T>()];

            var memory = new Memory<T>(result);
            var mh = memory.Pin();
            //CudaDriverApi.CudaCall(CudaDriverApi.cuMemcpyDtoH(mh.Pointer, ref _dataPointer, (uint)size));

            CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMemcpy(mh.Pointer, _dataPointer, (uint)size, CudaMemCpyKind.cudaMemcpyDeviceToHost));
            mh.Dispose();
            return result;
        }

        unsafe void SetDataInternal<T>(T[] data)
           where T : struct
        {
            if (_fmt.Type == RTPBufferType.Host)
            {
                var span = new Span<T>(data);
                MemoryHelper.CopyFromManaged(ref span, _dataPointer, (uint)data.Length);
            }
            else
            {
                var memory = new Memory<T>(data);
                var mh = memory.Pin();
                CudaRunTimeApi.CudaCall(CudaRunTimeApi.cudaMemcpy(_dataPointer.ToPointer(), new IntPtr(mh.Pointer), (uint)(data.Length * Unsafe.SizeOf<T>()), CudaMemCpyKind.cudaMemcpyHostToDevice));
                mh.Dispose();
            }
        }

        public T[] GetData<T>()
        {
            return GetDataInternal<T>();
        }

        public void SetData<T>(T[] data)
            where T : struct
        {
            SetDataInternal<T>(data);
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