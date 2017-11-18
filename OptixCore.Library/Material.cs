using System;
using System.Numerics;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Material : OptixNode, IVariableContainer
    {
        SurfaceProgramCollection mCollection;

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

        public override void Validate()
        {
            CheckError(Api.rtMaterialValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtMaterialDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
        }


        public int VariableCount
        {
            get
            {
                CheckError(Api.rtMaterialGetVariableCount(InternalPtr, out var count));
                return (int)count;
            }
        }

        public Variable this[int index]
        {
            get {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtMaterialGetVariable(InternalPtr, (uint)index, out var rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtMaterialGetVariable(InternalPtr, (uint)index, out var rtVar));

                if (value == null)
                {
                    CheckError(Api.rtMaterialRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    throw new OptixException("Material Error: Variable copying not yet implemented");
                }
            }
        }

        public Variable this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Material Error: Variable name is null or empty");

                CheckError(Api.rtMaterialQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar == IntPtr.Zero)
                    CheckError(Api.rtMaterialDeclareVariable(InternalPtr, name, out rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Material Error: Variable name is null or empty");

                CheckError(Api.rtMaterialQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar != IntPtr.Zero && value == null)
                {
                    CheckError(Api.rtMaterialRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    if (string.IsNullOrEmpty(name))
                        throw new OptixException("Material Error: Variable copying not yet implemented");
                }
            }
        }

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
                throw new ArgumentNullException("program", "Material Error: program cannot be null!");

            if (rayTypeIndex >= mContext.RayTypeCount)
                throw new ArgumentOutOfRangeException("rayTypeIndex", "rayTypeIndex cannot exceed the RayTypeCount set on the Context");

            if (program.RayType == RayHitType.Any)
                CheckError(Api.rtMaterialSetAnyHitProgram(InternalPtr, (uint)rayTypeIndex, program.InternalPtr));
            else
                CheckError(Api.rtMaterialSetClosestHitProgram(InternalPtr, (uint)rayTypeIndex, program.InternalPtr));
        }

 
    }
}