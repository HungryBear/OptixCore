using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Native
{
    internal class GlInterop
    {
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtBufferCreateFromGLBO(IntPtr context, uint bufferdesc, uint glId, ref IntPtr buffer);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtBufferGLRegister(IntPtr buffer);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtBufferGLUnregister(IntPtr buffer);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtBufferGetGLBOId(IntPtr buffer, out uint glId);
    }
}