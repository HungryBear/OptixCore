using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Geometry : VariableContainerNode
    {
        public Geometry(Context context) : base(context)
        {
            CheckError(Api.rtGeometryCreate(context.InternalPtr, ref InternalPtr));
        }

        internal Geometry(Context context, IntPtr geom) : base(context)
        {
            InternalPtr = geom;
        }

        protected override Func<RTresult> ValidateAction => () => Api.rtGeometryValidate(InternalPtr);
        protected override Func<RTresult> DestroyAction => () => Api.rtAccelerationDestroy(InternalPtr);
        protected override Func<int> GetVariableCount => () =>
         {
             CheckError(Api.rtGeometryGetVariableCount(InternalPtr, out var count));
             return (int)count;
         };

        protected override Func<int, IntPtr> GetVariable => index =>
        {
            CheckError(Api.rtGeometryGetVariable(InternalPtr, (uint)index, out var ptr));
            return ptr;
        };

        protected override Func<string, IntPtr> QueryVariable => name =>
        {
            CheckError(Api.rtGeometryQueryVariable(InternalPtr, name, out var ptr));
            return ptr;
        };

        protected override Func<string, IntPtr> DeclareVariable => name =>
        {
            CheckError(Api.rtGeometryDeclareVariable(InternalPtr, name, out var ptr));
            return ptr;
        };

        protected override Func<IntPtr, RTresult> RemoveVariable => ptr => Api.rtGeometryRemoveVariable(InternalPtr, ptr);

        /// <summary>
        /// Set the number of primitives for the geometry
        /// </summary>
        public uint PrimitiveCount
        {
            get
            {
                CheckError(Api.rtGeometryGetPrimitiveCount(InternalPtr, out var count));
                return count;
            }
            set => CheckError(Api.rtGeometrySetPrimitiveCount(InternalPtr, value));
        }

        /// <summary>
        /// Set the program that will run when a ray collides with geometry
        /// </summary>
        public OptixProgram IntersectionProgram
        {
            get
            {
                CheckError(Api.rtGeometryGetIntersectionProgram(InternalPtr, out var program));
                return new OptixProgram(mContext, program);
            }
            set => CheckError(Api.rtGeometrySetIntersectionProgram(InternalPtr, value.InternalPtr));
        }

        /// <summary>
        /// Set the bounding box program
        /// </summary>
        public OptixProgram BoundingBoxProgram
        {
            get
            {
                CheckError(Api.rtGeometryGetBoundingBoxProgram(InternalPtr, out var program));
                return new OptixProgram(mContext, program);
            }
            set => CheckError(Api.rtGeometrySetBoundingBoxProgram(InternalPtr, value.InternalPtr));
        }

    }
}