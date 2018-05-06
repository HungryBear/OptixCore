

using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Native
{    
    using RTsize = UInt64;

    internal unsafe class Api
    {

        [DllImport("optixu.1.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuGetSizeForRTformat(RTformat format, out uint size);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtGetVersion(ref uint version);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetDevices(IntPtr context, IntPtr devices);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, ref int p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, ref Int2 p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, ref uint p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, ref long p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, ref ulong p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, [MarshalAs(UnmanagedType.LPStr)]ref string str);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtDeviceGetDeviceCount(ref uint count);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1f(IntPtr v, float f1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2f(IntPtr v, float f1, float f2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3f(IntPtr v, float f1, float f2, float f3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4f(IntPtr v, float f1, float f2, float f3, float f4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1fv(IntPtr v, ref float f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2fv(IntPtr v, Vector2 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3fv(IntPtr v, Vector3 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4fv(IntPtr v, Vector4 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1i(IntPtr v, int i1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2i(IntPtr v, int i1, int i2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3i(IntPtr v, int i1, int i2, int i3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4i(IntPtr v, int i1, int i2, int i3, int i4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1iv(IntPtr v, ref int i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2iv(IntPtr v, ref Int2 i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3iv(IntPtr v, ref Int3 i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4iv(IntPtr v, int* i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1ui(IntPtr v, uint u1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2ui(IntPtr v, uint u1, uint u2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3ui(IntPtr v, uint u1, uint u2, uint u3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4ui(IntPtr v, uint u1, uint u2, uint u3, uint u4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet1uiv(IntPtr v, ref uint u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet2uiv(IntPtr v, uint* u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet3uiv(IntPtr v, ref UInt3 u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSet4uiv(IntPtr v, uint* u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix2x2fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix2x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix2x4fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix3x2fv(IntPtr v, int transpose, Matrix3x2 m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix3x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix3x4fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix4x2fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix4x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetMatrix4x4fv(IntPtr v, int transpose, Matrix4x4 m);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetObject(IntPtr v, IntPtr @object);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableSetUserData(IntPtr v, IntPtr size, IntPtr ptr);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1f(IntPtr v, ref float f1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2f(IntPtr v, ref float f1, ref float f2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3f(IntPtr v, ref float f1, ref float f2, ref float f3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4f(IntPtr v, ref float f1, ref float f2, ref float f3, ref float f4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1fv(IntPtr v, ref float f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2fv(IntPtr v, ref Vector2 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3fv(IntPtr v, ref Vector3 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4fv(IntPtr v, ref Vector4 f);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1i(IntPtr v, ref int i1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2i(IntPtr v, ref int i1, ref int i2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3i(IntPtr v, ref int i1, ref int i2, ref int i3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4i(IntPtr v, ref int i1, ref int i2, ref int i3, ref int i4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1iv(IntPtr v, ref int i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2iv(IntPtr v, ref Int2 i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3iv(IntPtr v, ref Int3 i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4iv(IntPtr v, int* i);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1ui(IntPtr v, ref uint u1);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2ui(IntPtr v, ref uint u1, ref uint u2);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3ui(IntPtr v, ref uint u1, ref uint u2, ref uint u3);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4ui(IntPtr v, ref uint u1, ref uint u2, ref uint u3, ref uint u4);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet1uiv(IntPtr v, uint* u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet2uiv(IntPtr v, uint* u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet3uiv(IntPtr v, ref UInt3 u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGet4uiv(IntPtr v, uint* u);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix2x2fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix2x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix2x4fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix3x2fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix3x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix3x4fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix4x2fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix4x3fv(IntPtr v, int transpose, float* m);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetMatrix4x4fv(IntPtr v, int transpose, ref Matrix4x4 m);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetObject(IntPtr v, IntPtr @object);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetUserData(IntPtr v, RTsize size, IntPtr ptr);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetName(IntPtr v, IntPtr name_return);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetAnnotation(IntPtr v, ref IntPtr annotation_return);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetType(IntPtr v, ref RTobjecttype type_return);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetContext(IntPtr v, IntPtr context);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtVariableGetSize(IntPtr v, ref RTsize size);

        // Context------------------------------------

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextCreate(ref IntPtr context);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextDestroy(IntPtr context);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextValidate(IntPtr context);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern void rtContextGetErrorString(IntPtr context, RTresult code, ref IntPtr return_string);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, IntPtr p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, int p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, uint p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, ulong p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, float p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, ref IntPtr p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, ref int p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, ref uint p);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetAttribute(IntPtr context, RTcontextattribute attrib, RTsize size, ref ulong p);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetDeviceCount(System.IntPtr context, ref uint count);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetStackSize(IntPtr context, RTsize stack_size_bytes);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetStackSize(IntPtr context, ref RTsize stack_size_bytes);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetTimeoutCallback(IntPtr context, IntPtr callback, double min_polling_seconds);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetEntryPointCount(IntPtr context, uint num_entry_points);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetEntryPointCount(IntPtr context, ref uint num_entry_points);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetRayGenerationProgram(IntPtr context, uint entry_point_index, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetRayGenerationProgram(IntPtr context, uint entry_point_index, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetExceptionProgram(IntPtr context, uint entry_point_index, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetExceptionProgram(IntPtr context, uint entry_point_index, IntPtr program);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetExceptionEnabled(IntPtr context, RTexception exception, int enabled);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetExceptionEnabled(IntPtr context, RTexception exception, ref int enabled);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetRayTypeCount(IntPtr context, uint num_ray_types);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetRayTypeCount(IntPtr context, ref uint num_ray_types);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetMissProgram(IntPtr context, uint ray_type_index, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetMissProgram(IntPtr context, uint ray_type_index, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetTextureSamplerFromId(IntPtr context, int sampler_id, IntPtr sampler);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextCompile(IntPtr context);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextLaunch1D(IntPtr context, uint entry_point_index, RTsize image_width);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextLaunch2D(IntPtr context, uint entry_point_index, RTsize image_width, RTsize image_height);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextLaunch3D(IntPtr context, uint entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetRunningState(IntPtr context, ref int running);

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextLaunchProgressive2D(IntPtr context, uint entry_index, RTsize width, RTsize height, uint max_subframes);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextStopProgressive(IntPtr context);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetPrintEnabled(IntPtr context, int enabled);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetPrintEnabled(IntPtr context, ref int enabled);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetPrintBufferSize(IntPtr context, RTsize buffer_size_bytes);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetPrintBufferSize(IntPtr context, ref RTsize buffer_size_bytes);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextSetPrintLaunchIndex(IntPtr context, int x, int y, int z);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetPrintLaunchIndex(IntPtr context, ref int x, ref int y, ref int z);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtContextDeclareVariable(IntPtr context, [MarshalAs(UnmanagedType.LPStr)]string name, out IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtContextQueryVariable(IntPtr context, [MarshalAs(UnmanagedType.LPStr)]string name, out IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextRemoveVariable(IntPtr context, IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetVariableCount(IntPtr context, ref uint count);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetVariable(IntPtr context, uint index, IntPtr v);


        // Program -------

        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramCreateFromPTXString(IntPtr context, [MarshalAs(UnmanagedType.LPStr)]string ptx, [MarshalAs(UnmanagedType.LPStr)]string program_name, out IntPtr program);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramCreateFromPTXFile(IntPtr context, [MarshalAs(UnmanagedType.LPStr)]string filename, [MarshalAs(UnmanagedType.LPStr)]string program_name, out IntPtr program);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramDestroy(IntPtr program);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramValidate(IntPtr program);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramGetContext(IntPtr program, IntPtr context);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramDeclareVariable(IntPtr program, [MarshalAs(UnmanagedType.LPStr)]string name, out IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramQueryVariable(IntPtr program, [MarshalAs(UnmanagedType.LPStr)]string name, out IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramRemoveVariable(IntPtr program, IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramGetVariableCount(IntPtr program, out uint count);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramGetVariable(IntPtr program, uint index, out IntPtr v);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtProgramGetId(IntPtr program, ref int program_id);
        [DllImport(OptixLibraries.OptixLib, CharSet = CharSet.Auto)]
        public static extern RTresult rtContextGetProgramFromId(IntPtr context, int program_id, IntPtr program);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupCreate")]
        public static extern RTresult rtGroupCreate(IntPtr context, ref IntPtr group);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupDestroy")]
        public static extern RTresult rtGroupDestroy(IntPtr group);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupValidate")]
        public static extern RTresult rtGroupValidate(IntPtr group);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupGetContext")]
        public static extern RTresult rtGroupGetContext(IntPtr group, ref IntPtr context);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupSetAcceleration")]
        public static extern RTresult rtGroupSetAcceleration(IntPtr group, IntPtr acceleration);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupGetAcceleration")]
        public static extern RTresult rtGroupGetAcceleration(IntPtr group, ref IntPtr acceleration);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupSetChildCount")]
        public static extern RTresult rtGroupSetChildCount(IntPtr group, uint count);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupGetChildCount")]
        public static extern RTresult rtGroupGetChildCount(IntPtr group, ref uint count);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupSetChild")]
        public static extern RTresult rtGroupSetChild(IntPtr group, uint index, IntPtr child);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupGetChild")]
        public static extern RTresult rtGroupGetChild(IntPtr group, uint index, ref IntPtr child);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGroupGetChildType")]
        public static extern RTresult rtGroupGetChildType(IntPtr group, uint index, ref RTobjecttype type);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorCreate")]
        public static extern RTresult rtSelectorCreate(IntPtr context, ref IntPtr selector);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorDestroy")]
        public static extern RTresult rtSelectorDestroy(IntPtr selector);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorValidate")]
        public static extern RTresult rtSelectorValidate(IntPtr selector);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetContext")]
        public static extern RTresult rtSelectorGetContext(IntPtr selector, ref IntPtr context);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorSetVisitProgram")]
        public static extern RTresult rtSelectorSetVisitProgram(IntPtr selector, IntPtr program);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetVisitProgram")]
        public static extern RTresult rtSelectorGetVisitProgram(IntPtr selector, ref IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorSetChildCount")]
        public static extern RTresult rtSelectorSetChildCount(IntPtr selector, uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetChildCount")]
        public static extern RTresult rtSelectorGetChildCount(IntPtr selector, ref uint count);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorSetChild")]
        public static extern RTresult rtSelectorSetChild(IntPtr selector, uint index, IntPtr child);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetChild")]
        public static extern RTresult rtSelectorGetChild(IntPtr selector, uint index, ref IntPtr child);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetChildType")]
        public static extern RTresult rtSelectorGetChildType(IntPtr selector, uint index, ref RTobjecttype type);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorDeclareVariable")]
        public static extern RTresult rtSelectorDeclareVariable(IntPtr selector, [In] [MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorQueryVariable")]
        public static extern RTresult rtSelectorQueryVariable(IntPtr selector, [In] [MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorRemoveVariable")]
        public static extern RTresult rtSelectorRemoveVariable(IntPtr selector, IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetVariableCount")]
        public static extern RTresult rtSelectorGetVariableCount(IntPtr selector, ref uint count);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtSelectorGetVariable")]
        public static extern RTresult rtSelectorGetVariable(IntPtr selector, uint index, ref IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformCreate")]
        public static extern RTresult rtTransformCreate(IntPtr context, ref IntPtr transform);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformDestroy")]
        public static extern RTresult rtTransformDestroy(IntPtr transform);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformValidate")]
        public static extern RTresult rtTransformValidate(IntPtr transform);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformGetContext")]
        public static extern RTresult rtTransformGetContext(IntPtr transform, ref IntPtr context);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformSetMatrix")]
        public static extern RTresult rtTransformSetMatrix(IntPtr transform, int transpose, ref Matrix4x4 matrix, ref Matrix4x4 inverse_matrix);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformGetMatrix")]
        public static extern RTresult rtTransformGetMatrix(IntPtr transform, int transpose, out Matrix4x4 matrix, out Matrix4x4 inverse_matrix);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformSetChild")]
        public static extern RTresult rtTransformSetChild(IntPtr transform, IntPtr child);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformGetChild")]
        public static extern RTresult rtTransformGetChild(IntPtr transform, out IntPtr child);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTransformGetChildType")]
        public static extern RTresult rtTransformGetChildType(IntPtr transform, ref RTobjecttype type);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupCreate")]
        public static extern RTresult rtGeometryGroupCreate(IntPtr context, ref IntPtr geometrygroup);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupDestroy")]
        public static extern RTresult rtGeometryGroupDestroy(IntPtr geometrygroup);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupValidate")]
        public static extern RTresult rtGeometryGroupValidate(IntPtr geometrygroup);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupGetContext")]
        public static extern RTresult rtGeometryGroupGetContext(IntPtr geometrygroup, ref IntPtr context);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupSetAcceleration")]
        public static extern RTresult rtGeometryGroupSetAcceleration(IntPtr geometrygroup, IntPtr acceleration);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupGetAcceleration")]
        public static extern RTresult rtGeometryGroupGetAcceleration(IntPtr geometrygroup, out IntPtr acceleration);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupSetChildCount")]
        public static extern RTresult rtGeometryGroupSetChildCount(IntPtr geometrygroup, uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupGetChildCount")]
        public static extern RTresult rtGeometryGroupGetChildCount(IntPtr geometrygroup, out uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupSetChild")]
        public static extern RTresult rtGeometryGroupSetChild(IntPtr geometrygroup, uint index, IntPtr geometryinstance);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGroupGetChild")]
        public static extern RTresult rtGeometryGroupGetChild(IntPtr geometrygroup, uint index, out IntPtr geometryinstance);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationCreate")]
        public static extern RTresult rtAccelerationCreate(IntPtr context, ref IntPtr acceleration);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationDestroy")]
        public static extern RTresult rtAccelerationDestroy(IntPtr acceleration);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationValidate")]
        public static extern RTresult rtAccelerationValidate(IntPtr acceleration);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetContext")]
        public static extern RTresult rtAccelerationGetContext(IntPtr acceleration, ref IntPtr context);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationSetBuilder")]
        public static extern RTresult rtAccelerationSetBuilder(IntPtr acceleration, [In] [MarshalAs(UnmanagedType.LPStr)] string builder);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetBuilder")]
        public static extern RTresult rtAccelerationGetBuilder(IntPtr acceleration, [MarshalAs(UnmanagedType.LPStr)] out string return_string);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationSetTraverser")]
        public static extern RTresult rtAccelerationSetTraverser(IntPtr acceleration, [In] [MarshalAs(UnmanagedType.LPStr)] string traverser);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetTraverser")]
        public static extern RTresult rtAccelerationGetTraverser(IntPtr acceleration, [MarshalAs(UnmanagedType.LPStr)] out string return_string);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationSetProperty")]
        public static extern RTresult rtAccelerationSetProperty(IntPtr acceleration, [In] [MarshalAs(UnmanagedType.LPStr)] string name, [In] [MarshalAs(UnmanagedType.LPStr)] string value);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetProperty")]
        public static extern RTresult rtAccelerationGetProperty(IntPtr acceleration, [In] [MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr return_string);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetDataSize")]
        public static extern RTresult rtAccelerationGetDataSize(IntPtr acceleration, ref uint size);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationGetData")]
        public static extern RTresult rtAccelerationGetData(IntPtr acceleration, IntPtr data);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationSetData")]
        public static extern RTresult rtAccelerationSetData(IntPtr acceleration, IntPtr data, uint size);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationMarkDirty")]
        public static extern RTresult rtAccelerationMarkDirty(IntPtr acceleration);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtAccelerationIsDirty")]
        public static extern RTresult rtAccelerationIsDirty(IntPtr acceleration, out int dirty);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceCreate")]
        public static extern RTresult rtGeometryInstanceCreate(IntPtr context, ref IntPtr geometryinstance);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceDestroy")]
        public static extern RTresult rtGeometryInstanceDestroy(IntPtr geometryinstance);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceValidate")]
        public static extern RTresult rtGeometryInstanceValidate(IntPtr geometryinstance);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetContext")]
        public static extern RTresult rtGeometryInstanceGetContext(IntPtr geometryinstance, ref IntPtr context);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceSetGeometry")]
        public static extern RTresult rtGeometryInstanceSetGeometry(IntPtr geometryinstance, IntPtr geometry);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetGeometry")]
        public static extern RTresult rtGeometryInstanceGetGeometry(IntPtr geometryinstance, out IntPtr geometry);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceSetMaterialCount")]
        public static extern RTresult rtGeometryInstanceSetMaterialCount(IntPtr geometryinstance, uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetMaterialCount")]
        public static extern RTresult rtGeometryInstanceGetMaterialCount(IntPtr geometryinstance, out uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceSetMaterial")]
        public static extern RTresult rtGeometryInstanceSetMaterial(IntPtr geometryinstance, uint index, IntPtr material);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetMaterial")]
        public static extern RTresult rtGeometryInstanceGetMaterial(IntPtr geometryinstance, uint index, out IntPtr material);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceDeclareVariable")]
        public static extern RTresult rtGeometryInstanceDeclareVariable(IntPtr geometryinstance, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceQueryVariable")]
        public static extern RTresult rtGeometryInstanceQueryVariable(IntPtr geometryinstance, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceRemoveVariable")]
        public static extern RTresult rtGeometryInstanceRemoveVariable(IntPtr geometryinstance, IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetVariableCount")]
        public static extern RTresult rtGeometryInstanceGetVariableCount(IntPtr geometryinstance, out uint count);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryInstanceGetVariable")]
        public static extern RTresult rtGeometryInstanceGetVariable(IntPtr geometryinstance, uint index, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryCreate")]
        public static extern RTresult rtGeometryCreate(IntPtr context, ref IntPtr geometry);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryDestroy")]
        public static extern RTresult rtGeometryDestroy(IntPtr geometry);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryValidate")]
        public static extern RTresult rtGeometryValidate(IntPtr geometry);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetContext")]
        public static extern RTresult rtGeometryGetContext(IntPtr geometry, ref IntPtr context);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometrySetPrimitiveCount")]
        public static extern RTresult rtGeometrySetPrimitiveCount(IntPtr geometry, uint num_primitives);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetPrimitiveCount")]
        public static extern RTresult rtGeometryGetPrimitiveCount(IntPtr geometry, out uint num_primitives);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometrySetPrimitiveIndexOffset")]
        public static extern RTresult rtGeometrySetPrimitiveIndexOffset(IntPtr geometry, uint index_offset);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetPrimitiveIndexOffset")]
        public static extern RTresult rtGeometryGetPrimitiveIndexOffset(IntPtr geometry, out uint index_offset);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometrySetBoundingBoxProgram")]
        public static extern RTresult rtGeometrySetBoundingBoxProgram(IntPtr geometry, IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetBoundingBoxProgram")]
        public static extern RTresult rtGeometryGetBoundingBoxProgram(IntPtr geometry, out IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometrySetIntersectionProgram")]
        public static extern RTresult rtGeometrySetIntersectionProgram(IntPtr geometry, IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetIntersectionProgram")]
        public static extern RTresult rtGeometryGetIntersectionProgram(IntPtr geometry, out IntPtr program);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryDeclareVariable")]
        public static extern RTresult rtGeometryDeclareVariable(IntPtr geometry, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryQueryVariable")]
        public static extern RTresult rtGeometryQueryVariable(IntPtr geometry, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryRemoveVariable")]
        public static extern RTresult rtGeometryRemoveVariable(IntPtr geometry, IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetVariableCount")]
        public static extern RTresult rtGeometryGetVariableCount(IntPtr geometry, out uint count);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtGeometryGetVariable")]
        public static extern RTresult rtGeometryGetVariable(IntPtr geometry, uint index, out IntPtr v);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialCreate")]
        public static extern RTresult rtMaterialCreate(IntPtr context, ref IntPtr material);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialDestroy")]
        public static extern RTresult rtMaterialDestroy(IntPtr material);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialValidate")]
        public static extern RTresult rtMaterialValidate(IntPtr material);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialGetContext")]
        public static extern RTresult rtMaterialGetContext(IntPtr material, ref IntPtr context);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialSetClosestHitProgram")]
        public static extern RTresult rtMaterialSetClosestHitProgram(IntPtr material, uint ray_type_index, IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialGetClosestHitProgram")]
        public static extern RTresult rtMaterialGetClosestHitProgram(IntPtr material, uint ray_type_index, out IntPtr program);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialSetAnyHitProgram")]
        public static extern RTresult rtMaterialSetAnyHitProgram(IntPtr material, uint ray_type_index, IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialGetAnyHitProgram")]
        public static extern RTresult rtMaterialGetAnyHitProgram(IntPtr material, uint ray_type_index, out IntPtr program);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialDeclareVariable")]
        public static extern RTresult rtMaterialDeclareVariable(IntPtr material, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialQueryVariable")]
        public static extern RTresult rtMaterialQueryVariable(IntPtr material, [In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialRemoveVariable")]
        public static extern RTresult rtMaterialRemoveVariable(IntPtr material, IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialGetVariableCount")]
        public static extern RTresult rtMaterialGetVariableCount(IntPtr material, out uint count);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtMaterialGetVariable")]
        public static extern RTresult rtMaterialGetVariable(IntPtr material, uint index, out IntPtr v);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerCreate")]
        public static extern RTresult rtTextureSamplerCreate(IntPtr context, out IntPtr texturesampler);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerDestroy")]
        public static extern RTresult rtTextureSamplerDestroy(IntPtr texturesampler);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerValidate")]
        public static extern RTresult rtTextureSamplerValidate(IntPtr texturesampler);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetContext")]
        public static extern RTresult rtTextureSamplerGetContext(IntPtr texturesampler, ref IntPtr context);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetWrapMode")]
        public static extern RTresult rtTextureSamplerSetWrapMode(IntPtr texturesampler, uint dimension, RTwrapmode wrapmode);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetWrapMode")]
        public static extern RTresult rtTextureSamplerGetWrapMode(IntPtr texturesampler, uint dimension, ref RTwrapmode wrapmode);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetFilteringModes")]
        public static extern RTresult rtTextureSamplerSetFilteringModes(IntPtr texturesampler, RTfiltermode minification, RTfiltermode magnification, RTfiltermode mipmapping);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetFilteringModes")]
        public static extern RTresult rtTextureSamplerGetFilteringModes(IntPtr texturesampler, ref RTfiltermode minification, ref RTfiltermode magnification, ref RTfiltermode mipmapping);



        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetMaxAnisotropy")]
        public static extern RTresult rtTextureSamplerSetMaxAnisotropy(IntPtr texturesampler, float value);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetMaxAnisotropy")]
        public static extern RTresult rtTextureSamplerGetMaxAnisotropy(IntPtr texturesampler, ref float value);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetMipLevelClamp")]
        public static extern RTresult rtTextureSamplerSetMipLevelClamp(IntPtr texturesampler, float minLevel, float maxLevel);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetMipLevelClamp")]
        public static extern RTresult rtTextureSamplerGetMipLevelClamp(IntPtr texturesampler, ref float minLevel, ref float maxLevel);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetMipLevelBias")]
        public static extern RTresult rtTextureSamplerSetMipLevelBias(IntPtr texturesampler, float value);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetMipLevelBias")]
        public static extern RTresult rtTextureSamplerGetMipLevelBias(IntPtr texturesampler, ref float value);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetReadMode")]
        public static extern RTresult rtTextureSamplerSetReadMode(IntPtr texturesampler, RTtexturereadmode readmode);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetReadMode")]
        public static extern RTresult rtTextureSamplerGetReadMode(IntPtr texturesampler, ref RTtexturereadmode readmode);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetIndexingMode")]
        public static extern RTresult rtTextureSamplerSetIndexingMode(IntPtr texturesampler, RTtextureindexmode indexmode);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetIndexingMode")]
        public static extern RTresult rtTextureSamplerGetIndexingMode(IntPtr texturesampler, ref RTtextureindexmode indexmode);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerSetBuffer")]
        public static extern RTresult rtTextureSamplerSetBuffer(IntPtr texturesampler, uint deprecated0, uint deprecated1, IntPtr buffer);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetBuffer")]
        public static extern RTresult rtTextureSamplerGetBuffer(IntPtr texturesampler, uint deprecated0, uint deprecated1, ref IntPtr buffer);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtTextureSamplerGetId")]
        public static extern RTresult rtTextureSamplerGetId(IntPtr texturesampler, ref int texture_id);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferCreate")]
        public static extern RTresult rtBufferCreate(IntPtr context, uint bufferdesc, ref IntPtr buffer);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferDestroy")]
        public static extern RTresult rtBufferDestroy(IntPtr buffer);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferValidate")]
        public static extern RTresult rtBufferValidate(IntPtr buffer);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetContext")]
        public static extern RTresult rtBufferGetContext(IntPtr buffer, ref IntPtr context);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetFormat")]
        public static extern RTresult rtBufferSetFormat(IntPtr buffer, RTformat format);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetFormat")]
        public static extern RTresult rtBufferGetFormat(IntPtr buffer, ref RTformat format);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetElementSize")]
        public static extern RTresult rtBufferSetElementSize(IntPtr buffer, uint size_of_element);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetElementSize")]
        public static extern RTresult rtBufferGetElementSize(IntPtr buffer, ref uint size_of_element);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetSize1D")]
        public static extern RTresult rtBufferSetSize1D(IntPtr buffer, uint width);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetSize1D")]
        public static extern RTresult rtBufferGetSize1D(IntPtr buffer, ref uint width);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetSize2D")]
        public static extern RTresult rtBufferSetSize2D(IntPtr buffer, uint width, uint height);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetSize2D")]
        public static extern RTresult rtBufferGetSize2D(IntPtr buffer, ref uint width, ref uint height);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetSize3D")]
        public static extern RTresult rtBufferSetSize3D(IntPtr buffer, uint width, uint height, uint depth);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetSize3D")]
        public static extern RTresult rtBufferGetSize3D(IntPtr buffer, ref uint width, ref uint height, ref uint depth);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetMipLevelSize1D")]
        public static extern RTresult rtBufferGetMipLevelSize1D(IntPtr buffer, uint level, ref uint width);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetMipLevelSize2D")]
        public static extern RTresult rtBufferGetMipLevelSize2D(IntPtr buffer, uint level, ref uint width, ref uint height);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetMipLevelSize3D")]
        public static extern RTresult rtBufferGetMipLevelSize3D(IntPtr buffer, uint level, ref uint width, ref uint height, ref uint depth);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetSizev")]
        public static extern RTresult rtBufferSetSizev(IntPtr buffer, uint dimensionality, ref uint dims);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetSizev")]
        public static extern RTresult rtBufferGetSizev(IntPtr buffer, uint dimensionality, ref uint dims);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetDimensionality")]
        public static extern RTresult rtBufferGetDimensionality(IntPtr buffer, ref uint dimensionality);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetMipLevelCount")]
        public static extern RTresult rtBufferGetMipLevelCount(IntPtr buffer, ref uint level);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferMap")]
        public static extern RTresult rtBufferMap(IntPtr buffer, ref IntPtr user_pointer);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferUnmap")]
        public static extern RTresult rtBufferUnmap(IntPtr buffer);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferMapEx")]
        public static extern RTresult rtBufferMapEx(IntPtr buffer, uint map_flags, uint level, IntPtr user_owned, ref IntPtr optix_owned);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferUnmapEx")]
        public static extern RTresult rtBufferUnmapEx(IntPtr buffer, uint level);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetId")]
        public static extern RTresult rtBufferGetId(IntPtr buffer, ref int buffer_id);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtContextGetBufferFromId")]
        public static extern RTresult rtContextGetBufferFromId(IntPtr context, int buffer_id, ref IntPtr buffer);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetProgressiveUpdateReady")]
        public static extern RTresult rtBufferGetProgressiveUpdateReady(IntPtr buffer, ref int ready, ref uint subframe_count, ref uint max_subframes);
        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferBindProgressiveStream")]
        public static extern RTresult rtBufferBindProgressiveStream(IntPtr stream, IntPtr source);

        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferSetAttribute")]
        public static extern RTresult rtBufferSetAttribute(IntPtr buffer, RTbufferattribute attrib, uint size, IntPtr p);


        [DllImport(OptixLibraries.OptixLib, EntryPoint = "rtBufferGetAttribute")]
        public static extern RTresult rtBufferGetAttribute(IntPtr buffer, RTbufferattribute attrib, uint size, IntPtr p);

        [DllImport("optixu.1.dll", EntryPoint = "rtRemoteDeviceCreate")]
        public static extern RTresult rtRemoteDeviceCreate([In] [MarshalAs(UnmanagedType.LPStr)] string url, [In] [MarshalAs(UnmanagedType.LPStr)] string username, [In] [MarshalAs(UnmanagedType.LPStr)] string password, ref IntPtr remote_dev);



        [DllImport("optixu.1.dll", EntryPoint = "rtRemoteDeviceDestroy")]
        public static extern RTresult rtRemoteDeviceDestroy(IntPtr remote_dev);

        [DllImport("optixu.1.dll", EntryPoint = "rtRemoteDeviceGetAttribute")]
        public static extern RTresult rtRemoteDeviceGetAttribute(IntPtr remote_dev, RTremotedeviceattribute attrib, uint size, IntPtr p);


        [DllImport("optixu.1.dll", EntryPoint = "rtRemoteDeviceReserve")]
        public static extern RTresult rtRemoteDeviceReserve(IntPtr remote_dev, uint num_nodes, uint configuration);



        [DllImport("optixu.1.dll", EntryPoint = "rtRemoteDeviceRelease")]
        public static extern RTresult rtRemoteDeviceRelease(IntPtr remote_dev);



    }

}
