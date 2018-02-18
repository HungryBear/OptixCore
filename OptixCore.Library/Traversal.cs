﻿using System;
using System.Runtime.InteropServices;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library
{

    public class TraversalStream<T> : BufferStream
        where T : struct
    {
        public BufferStream Stream;

        public override long Length => Stream.Length / Marshal.SizeOf<T>();

        public TraversalStream(BufferStream source) : base(source)
        {
            this.Stream = source;
        }

        internal TraversalStream(IntPtr buffer, long sizeInBytes, bool canRead, bool canWrite, bool ownData) : base(buffer, sizeInBytes, canRead, canWrite, ownData)
        {
        }

        public void GetData(T[] results)
        {
            if (results == null)
                throw new ArgumentNullException("results", "Results array cannot be null.");

            if ((results.Length * Marshal.SizeOf<T>()) != Stream.Length)
                throw new ArgumentOutOfRangeException("results", "Results array must be able to hold entire TraversalStream");

            Stream.ReadRange(results, 0, results.Length);
        }
    }

    public enum QueryType
    {
        /// <summary>
        /// Perform an any hit ray intersection
        /// </summary>
        AnyHit		= RTPquerytype.RTP_QUERY_TYPE_ANY,

		/// <summary>
		/// Perform a closest hit ray intersection
		/// </summary>
		ClosestHit	= RTPquerytype.RTP_QUERY_TYPE_CLOSEST,
	};

    public enum RtpBufferType
    {
        Host = RTPbuffertype.RTP_BUFFER_TYPE_HOST,
        Cuda = RTPbuffertype.RTP_BUFFER_TYPE_CUDA_LINEAR
    }

    public enum RtpBufferFormat
    {
        IndicesInt3 = 0x400, /*!< Index buffer with 3 integer vertex indices per triangle */
        IndicesInt3MaskInt = 0x401, /*!< Index buffer with 3 integer vertex indices per triangle, and an integer visibility mask */

        /* VERTICES */
        VERTEX_FLOAT3 = 0x420, /*!< Vertex buffer with 3 floats per vertex position */
        VERTEX_FLOAT4 = 0x421, /*!< Vertex buffer with 4 floats per vertex position */

        /* RAYS */
        RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION = 0x440, /*!< float3:origin float3:direction */
        RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX = 0x441, /*!< float3:origin, float:tmin, float3:direction, float:tmax */
        RTP_BUFFER_FORMAT_RAY_ORIGIN_MASK_DIRECTION_TMAX = 0x442, /*!< float3:origin, int:mask, float3:direction, float:tmax. If used, buffer format RTP_BUFFER_FORMAT_INDICES_INT3_MASK_INT is required! */

        /* HITS */
        RTP_BUFFER_FORMAT_HIT_BITMASK = 0x460, /*!< one bit per ray 0=miss, 1=hit */
        RTP_BUFFER_FORMAT_HIT_T = 0x461, /*!< float:ray distance (t < 0 for miss) */
        RTP_BUFFER_FORMAT_HIT_T_TRIID = 0x462, /*!< float:ray distance (t < 0 for miss), int:triangle id */
        RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V = 0x463, /*!< float:ray distance (t < 0 for miss), int:triangle id, float2:barycentric coordinates u,v (w=1-u-v) */

        RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID = 0x464, /*!< float:ray distance (t < 0 for miss), int:triangle id, int:instance position in list */
        RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V = 0x465, /*!< float:ray distance (t < 0 for miss), int:triangle id, int:instance position in list, float2:barycentric coordinates u,v (w=1-u-v) */

        /* INSTANCES */
        RTP_BUFFER_FORMAT_INSTANCE_MODEL = 0x480, /*!< RTPmodel:objects of type RTPmodel */

        /* TRANSFORM MATRICES */
        RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4 = 0x490, /*!< float:row major 4x4 affine matrix (it is assumed that the last row has the entries 0.0f, 0.0f, 0.0f, 1.0f, and will be ignored) */
        RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3 = 0x491  /*!< float:row major 4x3 affine matrix */
    }

    /// <summary>
    /// The input format of the ray vector.
    /// </summary>
    public enum RayFormat
    {
        /// <summary>
        /// Describes a ray format with interleaved origin, direction, tmin, and tmax
        /// e.g. ray {
        ///		float3 origin;
        ///		float3 dir;
        ///		float tmin;
        ///		float tmax;
        ///	}
        /// </summary>
        OriginDirectionMinMaxInterleaved	 = 0,

		/// <summary>
		/// Describes a ray format with interleaved origin, direction
		/// e.g. ray {
		///		float3 origin;
		///		float3 dir;
		///	}
		/// </summary>
		OriginDirectionInterleaved			 = 1,
	};

    /// <summary>
    /// The input format of the triangles.
    /// TriangleSoup implies future use of SetTriangles while Mesh implies use of SetMesh.
    /// </summary>
    public enum TriFormat
    {
        Mesh			= 0,
		TriangleSoup	= 1,
	};

    /// <summary>
    /// Initialization options (static across life of traversal object).
    ///
    /// The rtuTraverse API supports both running on the CPU and GPU.  When
    /// RTU_INITOPTION_NONE is specified GPU context creation is attempted.  If
    /// that fails (such as when there isn't an NVIDIA GPU part present, the CPU
    /// code path is automatically chosen.  Specifying RTU_INITOPTION_GPU_ONLY or
    /// RTU_INITOPTION_CPU_ONLY will only use the GPU or CPU modes without
    /// automatic transitions from one to the other.
    ///
    /// RTU_INITOPTION_CULL_BACKFACE will enable back face culling during
    /// intersection.
    /// </summary>
    [Flags]
    public enum InitOptions
    {
        None			= 0,
		GpuOnly			= 1<<0,
		CpuOnly			= 1 << 1,
		CullBackFace	= 1 << 2
    };

    public enum RunTimeOptions
    {
        NumThreads = 0
    };

    [Flags]
    public enum TraversalOutput
    {
        None		= 0,

		/// <summary>
		/// Outputs a float3 normal
		/// </summary>
		Normal		= 1<<0,
										 
		/// <summary>
		/// float2 [alpha, beta] (gamma implicit)
		/// </summary>
		BaryCentric	= 1<<1,

		/// <summary>
		///  char   [1 | 0]
		/// </summary>
		BackFacing	= 1<<2
    };

    /// <summary>
    /// Structure encapsulating the result of a single ray query.
    /// </summary>
    public struct TraversalResult
    {
		/// <summary>
		/// Index of the interesected triangle, -1 for miss.
		/// </summary>
		int PrimitiveID;

        /// <summary>
        /// Ray t parameter of hit point.
        /// </summary>
        float T;
    };

    public enum RTPBufferType
    {
        Host = 0x200,  /*!< Buffer in host memory */
        CudaLinear = 0x201   /*!< Linear buffer in device memory on a cuda device */
    };

    public abstract class OptixPrimeNode : IDisposable
    {
        public abstract void Validate();
        public abstract void Destroy();

        public virtual void CheckError(object result)
        {

        }

        protected internal PrimeContext mContext;
        protected internal IntPtr InternalPtr;

        protected OptixPrimeNode(PrimeContext context)
        {
            this.mContext = context;
        }

        ~OptixPrimeNode()
        {
            //Destroy();
        }

        internal void CheckError(RTPresult result)
        {
            if (result != RTPresult.RTP_SUCCESS)
            {
                PrimeApi.rtpGetErrorString(result, out var message);
                throw new OptixException($"Optix context error : {message}");
            }
        }

        public void Dispose()
        {
            Destroy();
            GC.SuppressFinalize(this);
        }
    }

    public class PrimeBufferDesc
    {
        public RtpBufferFormat Format { get; set; }
        public RTPBufferType Type { get; set; }

    }

    public class PrimeBuffer : OptixPrimeNode
    {
        public PrimeBuffer(PrimeContext context, PrimeBufferDesc fmt, IntPtr data) : base(context)
        {
            CheckError(PrimeApi.rtpBufferDescCreate(context.InternalPtr, (RTPbufferformat)fmt.Format, (RTPbuffertype)fmt.Type, data, out InternalPtr ));
        }

        public override void Validate()
        {
            
        }

        public override void Destroy()
        {
            CheckError(PrimeApi.rtpBufferDescDestroy(InternalPtr));
        }
    }

    public class PrimeModel : OptixPrimeNode
    {
        public PrimeModel(PrimeContext context) : base(context)
        {
        }

        public override void Validate()
        {
            throw new NotImplementedException();
        }

        public override void Destroy()
        {
            throw new NotImplementedException();
        }
    }

    public class PrimeContext : IDisposable
    {
        protected internal IntPtr InternalPtr;

        public PrimeContext()
        {
            CheckError(PrimeApi.rtpContextCreate(RTPcontexttype.RTP_CONTEXT_TYPE_CUDA, out InternalPtr));
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
                CheckError(PrimeApi.rtpContextDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }

        internal void CheckError(RTPresult result)
        {
            if (result != RTPresult.RTP_SUCCESS)
            {
                PrimeApi.rtpGetErrorString(result, out var message);
                throw new OptixException($"Optix context error : {message}");
            }
        }
    }
}