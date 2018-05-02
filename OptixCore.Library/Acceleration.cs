using System;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    /// <summary>
    /// Defines the Matrix layout used by Optix: Row major or Column major
    /// </summary>
    public enum MatrixLayout
    {
        ColumnMajor = 0,
        RowMajor = 1
    };

    /// <summary>
    /// Type of Bounding Volume Hierarchy algorithm that will be used for <see cref="OptixCore.Library.Acceleration">Acceleration</see> construction.
    /// </summary>
    public enum AccelBuilder
    {
        /// <summary>
        /// No Acceleration structure. Only suitable for very small scenes.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser">NoAccel</see>.
        /// </summary>
        NoAccel,

        /// <summary>
        /// Very fast GPU based accelertion good for animated scenes.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser">Bvh</see> or <see cref="OptixCore.Library.AccelTraverser">BvhCompact</see>.
        /// </summary>
        Lbvh,

        /// <summary>
        /// Classic Bounding Volume Hierarchy that favors quality over construction performance.<br/>
        /// A good compromise between Sbvh and MedianBvh.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser">Bvh</see> or <see cref="OptixCore.Library.AccelTraverser">BvhCompact</see>.
        /// </summary>
        Bvh,

        /// <summary>
        /// Uses a fast construction scheme to produce a medium quality bounding volume hierarchy.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser.Bvh">Bvh</see> or <see cref="OptixCore.Library.AccelTraverser.Bvh">BvhCompact</see>.
        /// </summary>
        MedianBvh,

        /// <summary>
        /// Split-BVH, a high quality bounding volume hierarchy that is usually the best choice for static geometry.<br/>
        /// Has the highest memory footprint and construction time.<br/>
        /// Requires certain properties to be set, such as: <see cref="OptixCore.Library.Acceleration.VertexBufferName">VertexBufferName</see>
        /// and <see cref="OptixCore.Library.Acceleration.IndexBufferName">IndexBufferName</see>.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser">Bvh</see> or <see cref="OptixCore.Library.AccelTraverser">BvhCompact</see>.
        /// </summary>
        Sbvh,

        /// <summary>
        /// Constructs a high quality kd-tree and is comparable to Sbvh in traversal perfromance.<br/>
        /// Has the highest memory footprint and construction time.<br/>
        /// Requires certain properties to be set, such as: <see cref="OptixCore.Library.Acceleration.VertexBufferName">VertexBufferName</see>
        /// and <see cref="OptixCore.Library.Acceleration.IndexBufferName">IndexBufferName</see>.<br/>
        /// Traversers: <see cref="OptixCore.Library.AccelTraverser">Bvh</see> or <see cref="OptixCore.Library.AccelTraverser">BvhCompact</see>.
        /// </summary>
        TriangleKdTree
    };

    /// <summary>
    /// Type of Bounding Volume Hierarchy algorithm that will be used for <see cref="OptixCore.Library.Acceleration">Acceleration</see> traversal.
    /// </summary>
    public enum AccelTraverser
    {
        /// <summary>
        /// No Acceleration structure. Linearly traverses primitives in the scene.<br/>
        /// Builders: <see cref="AccelBuilder">NoAccel</see>.
        /// </summary>
        NoAccel,

        /// <summary>
        /// Uses a classic Bounding Volume Hierarchy traversal.<br/>
        /// Builders: <see cref="OptixCore.Library.AccelBuilder">Lbvh</see>, <see cref="OptixCore.Library.AccelBuilder">Bvh</see>
        /// <see cref="OptixCore.Library.AccelBuilder">MedianBvh</see>, or <see cref="OptixCore.Library.AccelBuilder">Sbvh</see>.<br/>
        /// </summary>
        Bvh,

        /// <summary>
        /// Compresses bvh data by a factor of 4 before uploading to the GPU. Acceleration data is decompressed on the fly during traversal.<br/>
        /// Useful for large static scenes that require more than a gigabyte of memory, and to minimize page misses for virtual memory.<br/>
        /// Builders: <see cref="OptixCore.Library.AccelBuilder">Lbvh</see>, <see cref="OptixCore.Library.AccelBuilder">Bvh</see>
        /// <see cref="OptixCore.Library.AccelBuilder">MedianBvh</see>, or <see cref="OptixCore.Library.AccelBuilder">Sbvh</see>.<br/>
        /// </summary>
        BvhCompact,

        /// <summary>
        /// Uses a kd-tree traverser.<br/>
        /// Builders: <see cref="OptixCore.Library.AccelBuilder">TriangleKdTree</see>.
        /// </summary>
        KdTree
    };

    public class Acceleration : OptixNode
    {

        AccelBuilder mBuilder;
        AccelTraverser mTraverser;
        public Acceleration(Context context, AccelBuilder mBuilder, AccelTraverser mTraverser) : base(context)
        {
            CheckError(Api.rtAccelerationCreate(context.InternalPtr, ref InternalPtr));
            gch = GCHandle.Alloc(InternalPtr, GCHandleType.Pinned);

            Builder = mBuilder;
            Traverser = mTraverser;

            MarkAsDirty();
        }

        internal Acceleration(Context ctx, IntPtr acc) : base(ctx)
        {
            InternalPtr = acc;
            CheckError(Api.rtAccelerationGetBuilder(InternalPtr, out var builderStr));
            mBuilder = (AccelBuilder)Enum.Parse(mBuilder.GetType(), builderStr);

            CheckError(Api.rtAccelerationGetTraverser(InternalPtr, out var traverseStr));
            mTraverser = (AccelTraverser)Enum.Parse(mTraverser.GetType(), traverseStr);
        }

        public override void Validate()
        {
            CheckError(Api.rtAccelerationValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtAccelerationDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
            gch.Free();
        }

        public void MarkAsDirty()
        {
            CheckError(Api.rtAccelerationMarkDirty(InternalPtr));
        }

        public bool IsDirty()
        {
            CheckError(Api.rtAccelerationIsDirty(InternalPtr, out var dirty));
            return dirty == 1;
        }

        public BufferStream Data
        {
            get
            {
                uint size = 0u;
                CheckError(Api.rtAccelerationGetDataSize(InternalPtr, ref size));

                var data = IntPtr.Zero;
                CheckError(Api.rtAccelerationGetData(InternalPtr, data));

                return new BufferStream(data, size, true, false, true);
            }
            set => CheckError(Api.rtAccelerationSetData(InternalPtr, value.DataPointer, (uint)value.Length));
        }

        /// <summary>
        /// Sets the 'refit' property on the acceleration structure. Refit tells Optix that only small geometry changes have been made,
        /// and to NOT perform a full rebuild of the hierarchy. Only a valid property on Bvh built acceleration structures.
        /// </summary>
        public bool Refit
        {

            get
            {
                if (mBuilder == AccelBuilder.Bvh)
                {
                    IntPtr str = IntPtr.Zero;
                    CheckError(Api.rtAccelerationGetProperty(InternalPtr, "refit", ref str));

                    return int.Parse(Marshal.PtrToStringAnsi(str)) == 1;
                }

                return false;
            }
            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "refit", value ? "1" : "0"));
        }

        /// <summary>
        /// Sets 'vertex_buffer_name' property of the acceleration structure. This notifies Sbvh and TriangkeKdTree builders to look at the
        /// vertex buffer assigned to 'vertex_buffer_name' in order to build the hierarchy.
        /// This must match the name of the variable the buffer is attached to.
        /// Property only valid for Sbvh or TriangleKdTree built acceleration structures.
        /// </summary>
        public string VertexBufferName
        {
            get
            {
                var str = IntPtr.Zero;
                CheckError(Api.rtAccelerationGetProperty(InternalPtr, "vertex_buffer_name", ref str));

                return Marshal.PtrToStringAnsi(str);
            }
            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "vertex_buffer_name", value));
        }

        /// <summary>
        /// Sets 'vertex_buffer_stride' property in bytes of the acceleration structure. This defines the offset between two vertices. Default is assumed to be 0.
        /// Property only valid for Sbvh or TriangleKdTree built acceleration structures.
        /// </summary>
        public int VertexBufferStride
        {

            get
            {
                if (mBuilder == AccelBuilder.Sbvh || mBuilder == AccelBuilder.TriangleKdTree)
                {
                    var str = IntPtr.Zero;
                    CheckError(Api.rtAccelerationGetProperty(InternalPtr, "vertex_buffer_stride", ref str));
                    return int.Parse(Marshal.PtrToStringAnsi(str));
                }
                return 0;
            }

            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "vertex_buffer_stride", value.ToString()));
        }


        /// <summary>
        /// Sets 'index_buffer_name' property of the acceleration structure. This notifies Sbvh and TriangkeKdTree builders to look at the
        /// index buffer assigned to 'index_buffer_name' in order to build the hierarchy.
        /// This must match the name of the variable the buffer is attached to.
        /// Property only valid for Sbvh or TriangleKdTree built acceleration structures.
        /// </summary>
        public string IndexBufferName
        {
            get
            {
                var str = IntPtr.Zero;
                CheckError(Api.rtAccelerationGetProperty(InternalPtr, "index_buffer_name", ref str));

                return Marshal.PtrToStringAnsi(str);
            }
            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "index_buffer_name", value));
        }

        /// <summary>
        /// Sets 'index_buffer_stride' property in bytes of the acceleration structure. This defines the offset between two indices. Default is assumed to be 0.
        /// Property only valid for Sbvh or TriangleKdTree built acceleration structures.
        /// </summary>
        public int IndexBufferStride
        {

            get
            {
                if (mBuilder == AccelBuilder.Sbvh || mBuilder == AccelBuilder.TriangleKdTree)
                {
                    var str = IntPtr.Zero;
                    CheckError(Api.rtAccelerationGetProperty(InternalPtr, "index_buffer_stride", ref str));
                    return int.Parse(Marshal.PtrToStringAnsi(str));
                }
                return 0;
            }

            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "index_buffer_stride", value.ToString()));
        }

        /// <summary>
        /// Sets 'leaf_size' property of the acceleration structure.
        /// </summary>
        public int LeafSize
        {

            get
            {
                if (mBuilder == AccelBuilder.Sbvh || mBuilder == AccelBuilder.TriangleKdTree)
                {
                    var str = IntPtr.Zero;
                    CheckError(Api.rtAccelerationGetProperty(InternalPtr, "leaf_size", ref str));
                    return int.Parse(Marshal.PtrToStringAnsi(str));
                }
                return 0;
            }

            set => CheckError(Api.rtAccelerationSetProperty(InternalPtr, "leaf_size", value.ToString()));
        }

        public AccelTraverser Traverser
        {
            get => mTraverser;
            set
            {
                CheckError(Api.rtAccelerationSetTraverser(InternalPtr, value.ToString()));
                mTraverser = value;
            }
        }

        public AccelBuilder Builder
        {
            get => mBuilder;
            set
            {
                CheckError(Api.rtAccelerationSetBuilder(InternalPtr, value.ToString()));
                mBuilder = value;
            }
        }


    }
}