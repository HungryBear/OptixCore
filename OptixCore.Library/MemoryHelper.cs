﻿using System;
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

    }
}