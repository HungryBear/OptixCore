using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class OptixCommandList : DataNode
    {
        public OptixCommandList(Context context) : base(context)
        {
            Create();
        }

        public void FinalizeList()
        {
            CheckError(Api.rtCommandListFinalize(InternalPtr));
        }

        public void Execute()
        {
            CheckError(Api.rtCommandListExecute(InternalPtr));
        }

        public override void Validate()
        {
        }

        public void AppendPostprocessingStage(OptixPostprocessingStage stage, uint launchWidth, uint launchHeight)
        {
            CheckError(Api.rtCommandListAppendPostprocessingStage(InternalPtr, stage.InternalPtr, launchWidth, launchHeight));
        }

        public void AppendLaunch(uint entryPointIndex, uint launchWidth, uint launchHeight)
        {
            CheckError(Api.rtCommandListAppendLaunch2D(InternalPtr, entryPointIndex, launchWidth, launchHeight));
        }

        public override void Destroy()
        {
            CheckError(Api.rtCommandListDestroy(InternalPtr));
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }

        private void Create()
        {
            CheckError(Api.rtCommandListCreate(mContext.InternalPtr, ref InternalPtr));
        }
    }
}
