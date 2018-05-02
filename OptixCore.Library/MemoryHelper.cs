using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library
{
    public static unsafe class MemoryHelper
    {
        /// <summary>
        /// Provides the current address of the given element
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="t"></param>
        /// <returns></returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static System.IntPtr AddressOf<T>(T t)
            //refember ReferenceTypes are references to the CLRHeader
            //where TOriginal : struct
        {
            System.TypedReference reference = __makeref(t);

            return *(System.IntPtr*)(&reference);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        static System.IntPtr AddressOfRef<T>(ref T t)
            //refember ReferenceTypes are references to the CLRHeader
            //where TOriginal : struct
        {
            System.TypedReference reference = __makeref(t);

            System.TypedReference* pRef = &reference;

            return (System.IntPtr)pRef; //(&pRef)
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        public static void MemCopy(IntPtr dest, IntPtr src, uint count)
        {
            byte[] data = new byte[count];
            Marshal.Copy(src, data, 0, (int)count);
            Marshal.Copy(data, 0, dest, (int)count);
        }

        public static void BlitMemory(IntPtr src, IntPtr dst, uint sizeInBytes)
        {
            var srcSpan = new Span<byte>(src.ToPointer(), (int)sizeInBytes);
            var dstSpan = new Span<byte>(dst.ToPointer(), (int)sizeInBytes);
            int i = 0;
            while (i < sizeInBytes)
            {
                dstSpan[i] = srcSpan[i];
                i++;
            }
        }

        public static void CopyFromManaged<T>(ref Span<T> src, IntPtr dst, uint elements)
            where T:struct
        {
            var dstSpan = new Span<T>(dst.ToPointer(), (int)elements);
            int i = 0;
            while (i < elements)
            {
                dstSpan[i] = src[i];
                i++;
            }
        }

        public static void CopyFromUnmanaged<T>(IntPtr src, ref Span<T> dst, uint elements)
            where T : struct
        {
            var dstSpan = new Span<T>(src.ToPointer(), (int)elements);
            int i = 0;
            while (i < elements)
            {
                dst[i] = dstSpan[i];
                i++;
            }
        }
    }
}