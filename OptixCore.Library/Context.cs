using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public delegate int OptixTimeoutHandler();

    /// <summary>
    /// Instance of a running OptiX engine. Provides an interface for creating and launching the ray-tracing engine.
    /// </summary>
    public class Context : IDisposable, IVariableContainer
    {
        protected internal IntPtr InternalPtr;
        private GCHandle gch;
        /// <summary>
        /// Creates an OptiX context.
        /// </summary>
        public Context()
        {
            gch = GCHandle.Alloc(InternalPtr, GCHandleType.Pinned);
            CheckError(Api.rtContextCreate(ref InternalPtr));
            SetStackSize(1024uL);
        }


        public void SetStackSize(ulong size)
        {
            CheckError(Api.rtContextSetStackSize(InternalPtr, size));
        }

        /// <summary>
        /// Validates the current context. This will check the programs, buffers, and variables set to make sure everything links correctly.
        /// </summary>
        public void Validate()
        {
            CheckError(Api.rtContextValidate(InternalPtr));
        }

        /// <summary>
        /// Compiles the context. This will validate, compile programs, and build an acceleration structure.
        /// </summary>
        public void Compile()
        {
            Validate();
            CheckError(Api.rtContextCompile(InternalPtr));
        }

        /// <summary>
        /// Builds the acceleration tree for the geometry hierarchy. Internally uses a call to rtContextLaunch3D( 0, 0, 0, 0 ).
        /// This controls where the acceleration building happens. Otherwise the accel tree is built on the next optix launch.
        /// </summary>
        public void BuildAccelTree()
        {
            CheckError(Api.rtContextLaunch3D(InternalPtr, 0, 0, 0, 0));
        }

        /// <summary>
        /// Performs a 1D Optix Launch, using the entry point program at entryPointIndex
        /// </summary>
        public void Launch(uint entryPointIndex, ulong width)
        {
            CheckError(Api.rtContextLaunch1D(InternalPtr, entryPointIndex, width));
        }

        /// <summary>
        /// Performs a 2D Optix Launch, using the entry point program at entryPointIndex
        /// </summary>
        public void Launch(uint entryPointIndex, ulong width, ulong height)
        {
            CheckError(Api.rtContextLaunch2D(InternalPtr, entryPointIndex, width, height));
        }

        /// <summary>
        /// Performs a 3D Optix Launch, using the entry point program at entryPointIndex
        /// </summary>
        public void Launch(uint entryPointIndex, ulong width, ulong height, ulong depth)
        {
            CheckError(Api.rtContextLaunch3D(InternalPtr, entryPointIndex, width, height, depth));
        }

        public void SetOptixtTimeOutHandler(OptixTimeoutHandler optixTimeOut, double minPollingSeconds)
        {
            var funcPtr = Marshal.GetFunctionPointerForDelegate(optixTimeOut);
            CheckError(Api.rtContextSetTimeoutCallback(InternalPtr, funcPtr, minPollingSeconds));
        }

        public uint GetOptixVersion()
        {
            uint result = 0;
            CheckError(Api.rtGetVersion(ref result));

            return result;
        }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public uint RayTypeCount
        {
            get
            {
                var numRayTypes = 0u;
                CheckError(Api.rtContextGetRayTypeCount(InternalPtr, ref numRayTypes));
                return numRayTypes;
            }
            set => CheckError(Api.rtContextSetRayTypeCount(InternalPtr, value));
        }

        public uint GetDevicesCount()
        {
            uint count = 0;
            CheckError(Api.rtContextGetDeviceCount(InternalPtr, ref count));
            return count;
        }

        /// <summary>
        /// Gets or Sets the number of CPU threads Optix can use
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public int CpuThreadsCount
        {
            get
            {
                var numThreads = 0;
                CheckError(Api.rtContextGetAttribute(InternalPtr,
                    RTcontextattribute.RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(int), ref numThreads));
                return numThreads;
            }
            set => CheckError(Api.rtContextSetAttribute(InternalPtr,
                RTcontextattribute.RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(int), value));
        }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsGpuPagingEnabled
        {
            get
            {
                var active = 0;
                CheckError(Api.rtContextGetAttribute(InternalPtr,
                    RTcontextattribute.RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(int), ref active));
                return active == 1;
            }
        }
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsGpuPagingForcedlyDisabled
        {
            get
            {
                var active = 0;
                CheckError(Api.rtContextGetAttribute(InternalPtr,
                    RTcontextattribute.RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(int), ref active));
                return active == 1;
            }
        }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public int MaxTextureCount
        {
            get
            {
                var active = 0;
                CheckError(Api.rtContextGetAttribute(InternalPtr,
                    RTcontextattribute.RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(int), ref active));
                return active;
            }
        }
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ulong AvailableMemory
        {
            get
            {
                ulong active = 0;
                CheckError(Api.rtContextGetAttribute(InternalPtr,
                    RTcontextattribute.RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(ulong), ref active));
                return active;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
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
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(Api.rtContextDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
                gch.Free();
            }
        }

        protected void CheckError(RTresult result)
        {
            if (result != RTresult.RT_SUCCESS)
            {
                var message = IntPtr.Zero;
                Api.rtContextGetErrorString(InternalPtr, result, ref message);
                throw new OptixException($"Optix context error : {Marshal.PtrToStringAnsi(message)}");
            }
        }

        public uint EntryPointCount
        {
            get
            {
                var numRayTypes = 0u;
                CheckError(Api.rtContextGetEntryPointCount(InternalPtr, ref numRayTypes));
                return numRayTypes;
            }
            set => CheckError(Api.rtContextSetEntryPointCount(InternalPtr, value));
        }

        public bool EnableAllExceptions
        {
            set => CheckError(
                Api.rtContextSetExceptionEnabled(InternalPtr, RTexception.RT_EXCEPTION_ALL, value ? 1 : 0));
        }

        public int VariableCount
        {
            get
            {
                var count = 0u;
                CheckError(Api.rtContextGetVariableCount(InternalPtr, ref count));
                return (int)count;
            }
        }

        public bool PrintingEnabled
        {
            get
            {
                var enabled = 0;
                CheckError(Api.rtContextGetPrintEnabled(InternalPtr, ref enabled));
                return enabled == 1;
            }
            set
            {
                var on = value ? 1 : 0;
                CheckError(Api.rtContextSetPrintEnabled(InternalPtr, on));
            }
        }

        /// <summary>
        /// Gets or Sets the size of the buffer in bytes Optix will use for printing from Cuda programs
        /// </summary>
        public UInt64 PrintBufferSize
        {
            get
            {
                var size = 0uL;
                CheckError(Api.rtContextGetPrintBufferSize(InternalPtr, ref size));
                return size;
            }
            set => CheckError(Api.rtContextSetPrintBufferSize(InternalPtr, value));
        }

        /// <summary>
        /// Gets the running state of the Optix context. Returns true if rtContextLaunch is currently active for this Context.
        /// </summary>
        public bool Running
        {
            get
            {
                var running = 0;
                CheckError(Api.rtContextGetRunningState(InternalPtr, ref running));
                return running == 1;
            }
        }

        /// <summary>
        /// Sets the ray generation program at entry point index [index] on the optix context. Ray generation programs will trace the initial ray (e.g. the eye ray for a camera)
        /// </summary>
        /// <param name="entryPointIndex">Entry point index of the ray generation program</param>
        /// <param name="program">Ray generation program</param>
        public void SetRayGenerationProgram(uint entryPointIndex, OptixProgram program)
        {
            CheckError(Api.rtContextSetRayGenerationProgram(InternalPtr, entryPointIndex, program.InternalPtr));
        }

        /// <summary>
        /// Sets the ray miss program at entry point index [index] on the optix context. Miss programs handle what happens when rays do not intersect any geometry
        /// There are a 1:1 ratio of miss programs and ray types.
        /// </summary>
        /// <param name="index">Ray type index of the ray miss program</param>
        /// <param name="program">Ray generation program</param>
        public void SetRayMissProgram(uint index, OptixProgram program)
        {
            CheckError(Api.rtContextSetMissProgram(InternalPtr, index, program.InternalPtr));
        }

        /// <summary>
        /// Sets the ray generation program at entry point index [index] on the optix context. Exception programs handle exceptions during a context launch
        /// </summary>
        /// <param name="index">Entry point index of the exception program</param>
        /// <param name="program">Exception program</param>
        public void SetExceptionProgram(uint index, OptixProgram program)
        {
            CheckError(Api.rtContextSetExceptionProgram(InternalPtr, index, program.InternalPtr));
        }


        /*	
        /// <summary>
        /// Sets the launch dimension to limit output from rtPrintf to a specific launch index
        /// </summary>
        /// <param name="x">The launch index in the x dimension to which to limit the output of rtPrintf invocations. 
        /// If set to -1, output is generated for all launch indices in the x dimension.</param>
        /// <param name="y">The launch index in the y dimension to which to limit the output of rtPrintf invocations. 
        /// If set to -1, output is generated for all launch indices in the y dimension.</param>
        /// <param name="z">The launch index in the z dimension to which to limit the output of rtPrintf invocations. 
        /// If set to -1, output is generated for all launch indices in the z dimension.</param>
        property OptixDotNet::Math::Int3 PrintLaunchIndex
        {
            OptixDotNet::Math::Int3 get()
            {
                int x, y, z;
                CheckError( rtContextGetPrintLaunchIndex( mContext, &x, &y, &z ) );

                return OptixDotNet::Math::Int3( x, y, z );
            }

            void set( OptixDotNet::Math::Int3 index )
            {
                CheckError( rtContextSetPrintLaunchIndex( mContext, index.X, index.Y, index.Z ) );
            }
        }


   
    generic< typename V >
    Geometry^ Context::CreateClusteredMesh(	unsigned int usePTX32InHost64, 
                                            unsigned int numTris,
                                            array<V>^ vertices,
                                            array<unsigned int>^ indices,
                                            array<unsigned int>^ materialIndices )
    {
        if( indices->Length != numTris * 3 )
            throw gcnew ArgumentOutOfRangeException( "indices", "indices.Length not equal to NumTris * 3" );

        if( indices->Length != materialIndices->Length )
            throw gcnew ArgumentOutOfRangeException( "indices/materialIndices", "indices.Length must equal materialIndices.Length" );

        pin_ptr<V> vPtr = &vertices[0];
        pin_ptr<unsigned int> iPtr = &indices[0];
        pin_ptr<unsigned int> mPtr = &materialIndices[0];

        RTgeometry geom;
        rtuCreateClusteredMesh( mContext, usePTX32InHost64, &geom, vertices->Length, (float*)vPtr, numTris, iPtr, mPtr );

        return gcnew Geometry( this, geom );
    }

    generic< typename V >
    Geometry^ Context::CreateClusteredMesh(	unsigned int usePTX32InHost64, 
                                            unsigned int numTris,
                                            array<V>^ vertices,
                                            OptixBuffer^ normals,
                                            OptixBuffer^ texcoords,
                                            array<unsigned int>^ indices,
                                            array<unsigned int>^ normalIndices,
                                            array<unsigned int>^ texcoordIndices,
                                            array<unsigned int>^ materialIndices )
    {
        if( indices->Length != numTris * 3 )
            throw gcnew ArgumentOutOfRangeException( "indices", "indices.Length not equal to NumTris * 3" );

        if( indices->Length != materialIndices->Length )
            throw gcnew ArgumentOutOfRangeException( "indices/materialIndices", "indices.Length must equal materialIndices.Length" );

        pin_ptr<V> vPtr = &vertices[0];
        pin_ptr<unsigned int> iPtr = &indices[0];
        pin_ptr<unsigned int> nPtr = &indices[0];
        pin_ptr<unsigned int> tcPtr = &indices[0];
        pin_ptr<unsigned int> mPtr = &materialIndices[0];

        RTgeometry geom;
        rtuCreateClusteredMeshExt(	mContext, usePTX32InHost64, &geom, vertices->Length, (float*)vPtr, numTris, iPtr, mPtr, 
                                    normals->InternalPtr(), nPtr, 
                                    texcoords->InternalPtr(), tcPtr );

        return gcnew Geometry( this, geom );
    }   
             */

        public Variable this[int index]
        {
            get
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                var rtVar = IntPtr.Zero;

                CheckError(Api.rtContextGetVariable(InternalPtr, (uint)index, rtVar));

                return new Variable(this, rtVar);

            }
            set
            {

                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                var rtVar = IntPtr.Zero;

                CheckError(Api.rtContextGetVariable(InternalPtr, (uint)index, rtVar));

                if (value == null)
                {
                    CheckError(Api.rtContextRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    throw new OptixException("Context Error: Variable copying not yet implemented");
                }
            }
        }

        public Variable this[string name]
        {
            get
            {

                CheckError(Api.rtContextQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar == IntPtr.Zero)
                    CheckError(Api.rtContextDeclareVariable(InternalPtr, name, out rtVar));

                return new Variable(this, rtVar);
            }
            set
            {
                if (string.IsNullOrWhiteSpace(name))
                {
                    throw new ArgumentNullException(nameof(name));
                }
                CheckError(Api.rtContextQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar != IntPtr.Zero && value == null)
                {
                    CheckError(Api.rtContextRemoveVariable(InternalPtr, rtVar));

                }
                else
                {
                    throw new OptixException("Context Error: Variable copying not yet implemented");
                }

            }
        }

        public OptixDevice[] Devices
        {
            get
            {
                var devicesCount = GetDevicesCount();
                var indices = new int[devicesCount];
                var devicePtr = GCHandle.Alloc(indices[0]);
                //pin_ptr<int> devicePtr = &indices[0];

                CheckError(Api.rtContextGetDevices(InternalPtr, devicePtr.AddrOfPinnedObject()));

                var devices = new OptixDevice[devicesCount];
                for (var i = 0; i < devicesCount; i++)
                {
                    var device = new OptixDevice();
                    device.Index = indices[i];
                    CheckError(Api.rtDeviceGetAttribute(indices[i],
                        RTdeviceattribute.RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(int),
                        ref device.MaxThreadsPerBlock));
                    CheckError(Api.rtDeviceGetAttribute(indices[i], RTdeviceattribute.RT_DEVICE_ATTRIBUTE_CLOCK_RATE,
                        sizeof(int), ref device.ClockRate));
                    CheckError(Api.rtDeviceGetAttribute(indices[i],
                        RTdeviceattribute.RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(int),
                        ref device.ProcessorCount));
                    CheckError(Api.rtDeviceGetAttribute(indices[i],
                        RTdeviceattribute.RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(int),
                        ref device.ExecutionTimeOutEnabled));
                    CheckError(Api.rtDeviceGetAttribute(indices[i],
                        RTdeviceattribute.RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(int),
                        ref device.MaxHardwareTextureCount));
                    CheckError(Api.rtDeviceGetAttribute(indices[i], RTdeviceattribute.RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY,
                        sizeof(UInt64), ref device.TotalMemory));
                    CheckError(Api.rtDeviceGetAttribute(indices[i],
                        RTdeviceattribute.RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, (uint)Marshal.SizeOf<Int2>(),
                        ref device.ComputeCapability));

                    var name = string.Empty;
                    CheckError(Api.rtDeviceGetAttribute(indices[i], RTdeviceattribute.RT_DEVICE_ATTRIBUTE_NAME, 256,
                        ref name));
                    device.Name = name;

                    devices[i] = device;
                }
                devicePtr.Free();
                return devices;
            }

        }
    }
}