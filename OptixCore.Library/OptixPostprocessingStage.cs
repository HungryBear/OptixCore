using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class OptixPostprocessingStage : DataNode
    {
        public OptixPostprocessingStage(Context context, string name) : base(context)
        {
            Create(name);
        }

        public override void Validate()
        {
        }

        public override void Destroy()
        {
            CheckError(Api.rtPostProcessingStageDestroy(InternalPtr));

        }

        public Variable DeclareVariable(string name)
        {
            var varPtr = IntPtr.Zero;
            CheckError(Api.rtPostProcessingStageDeclareVariable(InternalPtr, name , ref varPtr));
            return new Variable(mContext, varPtr);
        }

        public Variable GetVariable(uint index)
        {
            var varPtr = IntPtr.Zero;
            CheckError(Api.rtPostProcessingStageGetVariable(InternalPtr, index, ref varPtr));
            return new Variable(mContext, varPtr);

        }

        public Variable QueryVariable(string name)
        {
            var varPtr = IntPtr.Zero;
            CheckError(Api.rtPostProcessingStageQueryVariable(InternalPtr, name, ref varPtr));
            return new Variable(mContext, varPtr);
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }

        private void Create(string name)
        {
            CheckError(Api.rtPostProcessingStageCreateBuiltin(mContext.InternalPtr, name, ref InternalPtr));
        }
    }
}