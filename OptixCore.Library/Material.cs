using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Material : VariableContainerNode
    {
        SurfaceProgramCollection mCollection;

        public SurfaceProgramCollection Programs => mCollection;

        public Material(Context context) : base(context)
        {
            CheckError(Api.rtMaterialCreate(context.InternalPtr, ref InternalPtr));

            mCollection = new SurfaceProgramCollection(this);
        }

        internal Material(Context ctx, IntPtr mMaterial) : base(ctx)
        {
            mCollection = new SurfaceProgramCollection(this);
            InternalPtr = mMaterial;
        }

        protected override Func<RTresult> ValidateAction => () => Api.rtMaterialValidate(InternalPtr);
        protected override Func<RTresult> DestroyAction => () => Api.rtMaterialDestroy(InternalPtr);

        protected override Func<int, IntPtr> GetVariable => index =>
        {
            CheckError(Api.rtMaterialGetVariable(InternalPtr, (uint)index, out var ptr));
            return ptr;
        };

        protected override Func<string, IntPtr> QueryVariable => name =>
        {
            CheckError(Api.rtMaterialQueryVariable(InternalPtr, name, out var ptr));
            return ptr;
        };
        protected override Func<string, IntPtr> DeclareVariable => name =>
        {
            CheckError(Api.rtMaterialDeclareVariable(InternalPtr, name, out var ptr));
            return ptr;
        };
        protected override Func<IntPtr, RTresult> RemoveVariable => ptr => Api.rtMaterialRemoveVariable(InternalPtr, ptr);

        protected override Func<int> GetVariableCount => () =>
        {
            CheckError(Api.rtMaterialGetVariableCount(InternalPtr, out var count));
            return (int)count;
        };

        public void SetProgram(string name, OptixProgram @object)
        {
            var var = this[name].InternalPtr;
            var pr = @object.InternalPtr;

            CheckError(Api.rtVariableSetObject(var, pr));
        }

        public SurfaceProgram GetSurfaceProgram(int rayTypeIndex)
        {
            if (rayTypeIndex > mContext.RayTypeCount)
            {
                throw new ArgumentOutOfRangeException("rayTypeIndex",
                    "rayTypeIndex cannot exceen the RayTypeCount set on the Context");
            }

            var type = RayHitType.Any;
            var result = Api.rtMaterialGetAnyHitProgram(InternalPtr, (uint)rayTypeIndex, out var program);

            if (result != RTresult.RT_SUCCESS)
            {
                type = RayHitType.Closest;
                result = Api.rtMaterialGetClosestHitProgram(InternalPtr, (uint)rayTypeIndex, out program);
            }

            CheckError(result);

            return new SurfaceProgram(mContext, type, program);
        }

        public void SetSurfaceProgram(int rayTypeIndex, SurfaceProgram program)
        {
            if (program == null)
                throw new ArgumentNullException("program", "DefaultMaterial Error: program cannot be null!");

            if (rayTypeIndex >= mContext.RayTypeCount)
                throw new ArgumentOutOfRangeException("rayTypeIndex", "rayTypeIndex cannot exceed the RayTypeCount set on the Context");

            if (program.RayType == RayHitType.Any)
                CheckError(Api.rtMaterialSetAnyHitProgram(InternalPtr, (uint)rayTypeIndex, program.InternalPtr));
            else
                CheckError(Api.rtMaterialSetClosestHitProgram(InternalPtr, (uint)rayTypeIndex, program.InternalPtr));
        }

 
    }
}