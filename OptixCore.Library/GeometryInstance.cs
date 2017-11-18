using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class GeometryInstance : OptixNode, IVariableContainer, IContainerNode<Material>, INodeCollectionProvider<Material>
    {
        protected NodeCollection<Material> mCollection;

        public NodeCollection<Material> Materials => mCollection;

        public uint MaterialCount
        {
            get
            {
                CheckError(Api.rtGeometryInstanceGetMaterialCount(InternalPtr, out var count));
                return count;
            }
            set => CheckError(Api.rtGeometryInstanceSetMaterialCount(InternalPtr, value));
        }

        public GeometryInstance(Context context) : base(context)
        {
            CheckError(Api.rtGeometryInstanceCreate(context.InternalPtr, ref InternalPtr));
            mCollection = new NodeCollection<Material>(this);
        }

        public GeometryInstance(Context context, IntPtr ptr) : base(context)
        {
            InternalPtr = ptr;
            mCollection = new NodeCollection<Material>(this);
        }

        public override void Validate()
        {
            CheckError(Api.rtGeometryInstanceValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtGeometryInstanceDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
        }
        public int ChildCount
        {
            get => (int)MaterialCount;
            set => MaterialCount = (uint)value;
        }

        Variable IVariableContainer.this[int index]
        {
            get
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtGeometryInstanceGetVariable(InternalPtr, (uint)index, out var rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                CheckError(Api.rtGeometryInstanceGetVariable(InternalPtr, (uint)index, out var rtVar));

                if (value == null)
                {
                    CheckError(Api.rtGeometryInstanceRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    throw new OptixException("GeometryInstance Error: Variable copying not yet implemented");
                }
            }
        }

        Variable IVariableContainer.this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("GeometryInstance Error: Variable name is null or empty");

                CheckError(Api.rtGeometryInstanceQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar == IntPtr.Zero)
                    CheckError(Api.rtGeometryInstanceDeclareVariable(InternalPtr, name, out rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("GeometryInstance Error: Variable name is null or empty");

                CheckError(Api.rtGeometryInstanceQueryVariable(InternalPtr, name, out var rtVar));

                if (rtVar != IntPtr.Zero && value == null)
                {
                    CheckError(Api.rtGeometryInstanceRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    if (string.IsNullOrEmpty(name))
                        throw new OptixException("GeometryInstance Error: Variable copying not yet implemented");
                }
            }
        }

        public int VariableCount
        {
            get
            {
                CheckError(Api.rtGeometryInstanceGetVariableCount(InternalPtr, out var count));
                return (int)count;
            }
        }

        public Geometry Geometry
        {
            get
            {
                CheckError(Api.rtGeometryInstanceGetGeometry(InternalPtr, out var geom));
                return new Geometry(mContext, geom);
            }
            set => CheckError(Api.rtGeometryInstanceSetGeometry(InternalPtr, value.InternalPtr));
        }

        public Material this[int index]
        {
            get => GetMaterial(index);
            set => SetMaterial(index, value);
        }


        public void RemoveMaterialAtIndex(int index)
        {
            if (MaterialCount > 0)
            {
                CheckError(Api.rtGeometryInstanceGetMaterial(InternalPtr, MaterialCount - 1, out var temp));
                CheckError(Api.rtGeometryInstanceSetMaterial(InternalPtr, (uint)index, temp));
                MaterialCount--;
            }
        }

        public void AddMaterial(Material material)
        {
            SetMaterial((int)MaterialCount++, material);
        }


        public bool RemoveMaterial(Material material )
        {
            if (MaterialCount == 0)
                return false;

            var mtrl = material.InternalPtr;

            for (uint i = 0; i < MaterialCount; i++)
            {
                CheckError(Api.rtGeometryInstanceGetMaterial(InternalPtr, i, out var temp));
                if (mtrl == temp)
                {
                    RemoveMaterialAtIndex((int)i);
                    return true;
                }
            }

            return false;
        }

        private void SetMaterial(int index, Material value)
        {
            CheckError(Api.rtGeometryInstanceSetMaterial(InternalPtr, (uint)index, value.InternalPtr));
        }

        private Material GetMaterial(int index)
        {
            if (index >= MaterialCount)
            {
                throw new ArgumentOutOfRangeException("index");
            }

            CheckError(Api.rtGeometryInstanceGetMaterial(InternalPtr, (uint)index, out var mtrl));

            return new Material(mContext, mtrl);
        }

        public NodeCollection<Material> Collection => mCollection;
    }
}