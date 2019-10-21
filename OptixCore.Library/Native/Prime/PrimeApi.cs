using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Native.Prime
{
    using RTPcontext = IntPtr;
    using RTPbufferdesc = IntPtr;
    using RTPmodel = IntPtr;
    using RTPquery = IntPtr;
    using RTPsize = UInt64;


    internal class PrimeApi
    {
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpContextCreate(RTPcontexttype type, out IntPtr context);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpContextSetCudaDeviceNumbers(RTPcontext context, uint deviceCount, uint deviceNumbers);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpContextSetCpuThreads(RTPcontext context, uint numThreads);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpContextDestroy(RTPcontext context);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpContextGetLastErrorString(RTPcontext context, out IntPtr returnString);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescCreate(RTPcontext context, RTPbufferformat format, RTPbuffertype type, IntPtr buffer, out RTPbufferdesc desc);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescSetRange(RTPbufferdesc desc, RTPsize begin, RTPsize end);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescGetContext(RTPbufferdesc desc, out RTPcontext context);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescSetStride(RTPbufferdesc desc, uint strideBytes);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescSetCudaDeviceNumber(RTPbufferdesc desc, uint deviceNumber);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpBufferDescDestroy(RTPbufferdesc desc);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelCreate(RTPcontext context, out RTPmodel model);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelGetContext(RTPmodel model, out RTPcontext context);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelSetTriangles(RTPmodel model, RTPbufferdesc indices, RTPbufferdesc vertices);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelSetInstances(RTPmodel model, RTPbufferdesc instances, RTPbufferdesc transforms);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelUpdate(RTPmodel model, uint hints);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelFinish(RTPmodel model);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelGetFinished(RTPmodel model, out int isFinished);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelCopy(RTPmodel model, RTPmodel srcModel);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelSetBuilderParameter(RTPmodel model_api, RTPbuilderparam param, RTPsize size, IntPtr ptr);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpModelDestroy(RTPmodel model);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryCreate(RTPmodel model, RTPquerytype queryType, out RTPquery query);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryGetContext(RTPquery query, out RTPcontext context);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQuerySetRays(RTPquery query, RTPbufferdesc rays);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQuerySetHits(RTPquery query, RTPbufferdesc hits);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryExecute(RTPquery query, uint hints);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryFinish(RTPquery query);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryGetFinished(RTPquery query, out int isFinished);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQuerySetCudaStream(RTPquery query, IntPtr stream);

        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpQueryDestroy(RTPquery query);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpHostBufferLock(IntPtr buffer, RTPsize size);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpHostBufferUnlock(IntPtr buffer);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpGetErrorString(RTPresult errorCode, out IntPtr errorString );
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpGetVersion( out uint version);
        [DllImport(OptixLibraries.OptixPrimeLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTPresult rtpGetVersionString(out IntPtr versionString );
    }

}