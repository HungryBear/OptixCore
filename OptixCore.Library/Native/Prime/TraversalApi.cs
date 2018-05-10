using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace OptixCore.Library.Native.Prime
{
    using RTUtraversal = IntPtr;
    using RTcontext = IntPtr;
    public enum QueryType
    {
        /// <summary>
        /// Perform an any hit ray intersection
        /// </summary>
        AnyHit = 0,

        /// <summary>
        /// Perform a closest hit ray intersection
        /// </summary>
        ClosestHit = 1,
    }
    public enum TriFormat
    {
        Mesh			= RTUtriformat.RTU_TRIFORMAT_MESH,
        TriangleSoup	= RTUtriformat.RTU_TRIFORMAT_TRIANGLE_SOUP,
    };

    [Flags]
    public enum InitOptions
    {
        None			= RTUinitoptions.RTU_INITOPTION_NONE,
        GpuOnly			= RTUinitoptions.RTU_INITOPTION_GPU_ONLY,
        CpuOnly			= RTUinitoptions.RTU_INITOPTION_CPU_ONLY,
        CullBackFace	= RTUinitoptions.RTU_INITOPTION_CULL_BACKFACE
    };

    public enum RunTimeOptions
    {
        NumThreads = RTUoption.RTU_OPTION_INT_NUM_THREADS
    };

    [Flags]
    public enum TraversalOutput
    {
        None		= RTUoutput.RTU_OUTPUT_NONE,

        /// <summary>
        /// Outputs a float3 normal
        /// </summary>
        Normal		= RTUoutput.RTU_OUTPUT_NORMAL,
										 
        /// <summary>
        /// float2 [alpha, beta] (gamma implicit)
        /// </summary>
        BaryCentric	= RTUoutput.RTU_OUTPUT_BARYCENTRIC,

        /// <summary>
        ///  char   [1 | 0]
        /// </summary>
        BackFacing	= RTUoutput.RTU_OUTPUT_BACKFACING
    };



    public enum RTUquerytype
    {
        RTU_QUERY_TYPE_ANY_HIT = 0,  /*!< Perform any hit calculation     */
        RTU_QUERY_TYPE_CLOSEST_HIT,  /*!< Perform closest hit calculation */
        RTU_QUERY_TYPE_COUNT         /*!< Query type count                */
    }

    public enum RTUrayformat
    {
        RTU_RAYFORMAT_ORIGIN_DIRECTION_TMIN_TMAX_INTERLEAVED = 0, /*!< Origin Direction Tmin Tmax interleaved            */
        RTU_RAYFORMAT_ORIGIN_DIRECTION_INTERLEAVED,               /*!< Origin Direction interleaved                      */
        RTU_RAYFORMAT_COUNT                                       /*!< Ray format count                                  */
    }

    public enum RTUtriformat
    {
        RTU_TRIFORMAT_MESH = 0,        /*!< Triangle format mesh     */
        RTU_TRIFORMAT_TRIANGLE_SOUP,  /*!< Triangle 'soup' format   */
        RTU_TRIFORMAT_COUNT           /*!< Triangle format count    */
    }
    public enum RTUinitoptions
    {
        RTU_INITOPTION_NONE = 0,       /*!< No option         */
        RTU_INITOPTION_GPU_ONLY = 1 << 0,  /*!< GPU only          */
        RTU_INITOPTION_CPU_ONLY = 1 << 1,  /*!< CPU only          */
        RTU_INITOPTION_CULL_BACKFACE = 1 << 2   /*!< Back face culling */
    }    
    public enum RTUoutput
    {
        RTU_OUTPUT_NONE = 0,      /*!< Output None */
        RTU_OUTPUT_NORMAL = 1 << 0, /*!< float3 [x, y, z]                      */
        RTU_OUTPUT_BARYCENTRIC = 1 << 1, /*!< float2 [alpha, beta] (gamma implicit) */
        RTU_OUTPUT_BACKFACING = 1 << 2  /*!< char   [1 | 0]                        */
    }
     
    public enum RTUoption
    {
        RTU_OPTION_INT_NUM_THREADS = 0  /*!< Number of threads */
    }

    public enum RayFormat
    {
        OriginDirectionMinMaxInterleaved,
        OriginDirectionInterleaved = 1
    }

    /// <summary>
    /// Structure encapsulating the result of a single ray query.
    /// </summary>
    public struct TraversalResult
    {
        /// <summary>
        /// Index of the interesected triangle, -1 for miss.
        /// </summary>
        public int PrimitiveID;

        /// <summary>
        /// Ray t parameter of hit point.
        /// </summary>
        public float T;

        public override string ToString()
        {
            return PrimitiveID == 0  && Math.Abs(T) < 1e-14f ? "Empty":
                $"Obj {PrimitiveID} - Dist {T}";
        }
    }

    internal class TraversalApi
    {
        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalCreate(ref RTUtraversal traversal,
            RTUquerytype query_type,
            RTUrayformat ray_format,
            RTUtriformat tri_format,
            uint outputs,
            uint options,
            RTcontext context);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalSetMesh(RTUtraversal traversal,
            uint num_verts,
            IntPtr verts,
            uint num_tris,
            IntPtr indices );

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalMapRays(RTUtraversal traversal, uint num_rays, ref IntPtr rays);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalUnmapRays(RTUtraversal traversal);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalTraverse(RTUtraversal traversal);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalMapResults(RTUtraversal traversal, ref IntPtr results);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalUnmapResults(RTUtraversal traversal);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalMapOutput(RTUtraversal traversal, RTUoutput which, ref IntPtr output);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalUnmapOutput(RTUtraversal traversal, RTUoutput which);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalGetErrorString(RTUtraversal traversal,RTresult code, [MarshalAs(UnmanagedType.LPStr)]out string  return_string);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalSetOption(RTUtraversal traversal, RunTimeOptions option, IntPtr value);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalDestroy(RTUtraversal traversal);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalPreprocess(RTUtraversal traversal);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalGetAccelDataSize(RTUtraversal traversal,ref uint data_size);

        [DllImport(OptixLibraries.OptixULib, CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        public static extern RTresult rtuTraversalGetAccelData(RTUtraversal traversal, IntPtr data);

    }
}
