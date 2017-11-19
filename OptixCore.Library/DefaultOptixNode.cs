using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public abstract class DefaultOptixNode : OptixNode
    {
        protected abstract Func<RTresult> ValidateAction { get; }
        protected abstract Func<RTresult> DestroyAction { get; }




        protected DefaultOptixNode(Context context) : base(context)
        {
        }


        public override void Validate()
        {
            CheckError(ValidateAction());
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(DestroyAction());
            InternalPtr = IntPtr.Zero;
        }


       
    }
}