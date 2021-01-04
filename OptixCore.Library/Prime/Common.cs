using System;
using System.Runtime.InteropServices;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{

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
        OriginDirectionMinMaxInterleaved = 0,

        /// <summary>
        /// Describes a ray format with interleaved origin, direction
        /// e.g. ray {
        ///		float3 origin;
        ///		float3 dir;
        ///	}
        /// </summary>
        OriginDirectionInterleaved = 1,
    };

    /// <summary>
    /// The input format of the triangles.
    /// TriangleSoup implies future use of SetTriangles while Mesh implies use of SetMesh.
    /// </summary>
    public enum TriFormat
    {
        Mesh = 0,
        TriangleSoup = 1,
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
        None = 0,
        GpuOnly = 1 << 0,
        CpuOnly = 1 << 1,
        CullBackFace = 1 << 2
    };

    public enum RunTimeOptions
    {
        NumThreads = 0
    };

    [Flags]
    public enum TraversalOutput
    {
        None = 0,

        /// <summary>
        /// Outputs a float3 normal
        /// </summary>
        Normal = 1 << 0,

        /// <summary>
        /// float2 [alpha, beta] (gamma implicit)
        /// </summary>
        BaryCentric = 1 << 1,

        /// <summary>
        ///  char   [1 | 0]
        /// </summary>
        BackFacing = 1 << 2
    };



    public enum RTPBufferType
    {
        Host = 0x200,  /*!< Buffer in host memory */
        CudaLinear = 0x201   /*!< Linear buffer in device memory on a cuda device */
    };


    public abstract class OptixPrimeNode : BasePrimeEntity, IDisposable
    {
        public abstract void Validate();
        public abstract void Destroy();

        protected internal PrimeContext mContext;

        protected OptixPrimeNode(PrimeContext context)
        {
            this.mContext = context;
        }

        internal new void CheckError(RTPresult result)
        {
            if (result != RTPresult.RTP_SUCCESS)
            {
                PrimeApi.rtpGetErrorString(result, out var Errormessage);
                throw new OptixException($"Optix context error : {Marshal.PtrToStringAnsi(Errormessage)}");
            }
        }


        public void Dispose()
        {
            Destroy();
            GC.SuppressFinalize(this);
        }
    }
}