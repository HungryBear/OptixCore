using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Geometry : OptixNode, IVariableContainer
    {
        public Geometry(Context context) : base(context)
        {
            CheckError(Api.rtGeometryCreate(context.InternalPtr, ref InternalPtr));
        }

        internal Geometry(Context context, IntPtr geom) : base(context)
        {
            InternalPtr = geom;
        }

        public override void Validate()
        {
            CheckError(Api.rtGeometryValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(Api.rtGeometryDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }
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

        public Variable this[int index]
        {
            get
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtGeometryGetVariable(InternalPtr, (uint)index, out var rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtGeometryGetVariable(InternalPtr, (uint)index, out var rtVar));

                if (value == null)
                {
                    CheckError(Api.rtGeometryRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    throw new OptixException("Geometry Error: Variable copying not yet implemented");
                }
            }
        }

        public Variable this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Geometry Error: Variable name is null or empty");

                CheckError(Api.rtGeometryQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar == IntPtr.Zero)
                    CheckError(Api.rtGeometryDeclareVariable(InternalPtr, name, out rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Geometry Error: Variable name is null or empty");

                CheckError(Api.rtGeometryQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar != IntPtr.Zero && value == null)
                {
                    CheckError(Api.rtGeometryRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    if (string.IsNullOrEmpty(name))
                        throw new OptixException("Geometry Error: Variable copying not yet implemented");
                }
            }
        }

        public int VariableCount
        {
            get
            {
                CheckError(Api.rtGeometryGetVariableCount(InternalPtr, out var count));
                return (int)count;
            }
        }
    }
}