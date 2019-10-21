using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Native
{
    public enum CudaResult : uint
    {
        Success = 0,
        ErrorInvalidValue = 1,
        ErrorMemoryAllocation = 2,
        ErrorInitializationError = 3,

    }

    public enum CudaMemCpyKind : uint
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4,
    }


    internal unsafe class CudaRunTimeApi
    {
        internal const string RunTimeAPIDll = "cudart64_101.dll";
        [DllImport(RunTimeAPIDll, EntryPoint = "cudaMalloc")]
        public static extern CudaResult cudaMalloc(ref IntPtr dptr, uint bytesize);

        [DllImport(RunTimeAPIDll, EntryPoint = "cudaFree")]
        public static extern CudaResult cudaFree(IntPtr devPtr);

        [DllImport(RunTimeAPIDll, EntryPoint = "cudaMemcpy")]
        public static extern CudaResult cudaMemcpy(void* dst, IntPtr src, uint bytesize, CudaMemCpyKind kind);

        [DllImport(RunTimeAPIDll, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl, EntryPoint = "cudaGetErrorString")]
        public static extern IntPtr cudaGetErrorString(CudaResult c);

        public static void CudaCall(CudaResult result)
        {
            if (result != CudaResult.Success)
            {
                var str = Marshal.PtrToStringAnsi(cudaGetErrorString(result));
                throw new ApplicationException(str);
            }
        }
    }
}