using System;
using System.Numerics;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Transform : ContainerNode
    {
        public MatrixLayout MatrixLayout { get; set; }

        public Matrix4x4 Matrix
        {
            get
            {
                CheckError(Api.rtTransformGetMatrix(InternalPtr, (int)MatrixLayout, out var matrix, out var mi));
                return matrix;
            }
            set
            {
                Matrix4x4.Invert(value, out var res);
                CheckError(Api.rtTransformSetMatrix(InternalPtr, (int)MatrixLayout, ref value, ref res));
            }
        }

        public ContainerNode Child
        {
            get
            {
                CheckError(Api.rtTransformGetChild(InternalPtr, out var @object));
                var type = RTobjecttype.RT_OBJECTTYPE_UNKNOWN;
                CheckError(Api.rtTransformGetChildType(InternalPtr, ref type));
                return ConstructContainerNodeFromType((OptixTypes)(int)type, @object);
            }

            set => CheckError(Api.rtTransformSetChild(InternalPtr, value.ObjectPtr()));
        }

        public Transform(Context context) : base(context)
        {
            CheckError(Api.rtTransformCreate(context.InternalPtr, ref InternalPtr));
        }

        internal Transform(Context context, IntPtr ptr) : base(context)
        {
            InternalPtr = ptr;
        }

        public override void Validate()
        {
            CheckError(Api.rtTransformValidate(InternalPtr));

        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(Api.rtTransformDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }
    }
}