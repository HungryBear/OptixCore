using System;
using System.Collections.Generic;
using OptixCore.Library.Native;
namespace OptixCore.Library
{
    public class GeometryGroup : ContainerNode, IContainerNode<GeometryInstance>, INodeCollectionProvider<GeometryInstance>
    {
        NodeCollection<GeometryInstance> mCollection;

        public GeometryGroup(Context context) : base(context)
        {
            CheckError(Api.rtGeometryGroupCreate(context.InternalPtr, ref InternalPtr));
            mCollection = new NodeCollection<GeometryInstance>(this);
        }

        internal GeometryGroup(Context context, IntPtr mGroup) : base(context)
        {
            InternalPtr = mGroup;
            mCollection = new NodeCollection<GeometryInstance>(this);
        }

        public override void Validate()
        {
            CheckError(Api.rtGeometryGroupValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtGeometryGroupDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }

        public int ChildCount
        {
            get
            {
                CheckError(Api.rtGeometryGroupGetChildCount(InternalPtr, out var count));
                return (int)count;
            }
            set => CheckError(Api.rtGeometryGroupSetChildCount(InternalPtr, (uint)value));
        }

        public int VariableCount
        {
            get
            {
                CheckError(Api.rtGeometryGroupGetChildCount(InternalPtr, out var count));
                return (int)count;
            }
        }

        public GeometryInstance this[int index]
        {
            get => GetChild(index);
            set => SetChild(index, value);
        }

        public Acceleration Acceleration
        {
            get
            {
                CheckError(Api.rtGeometryGroupGetAcceleration(InternalPtr, out var accel));

                return new Acceleration(mContext, accel);
            }
            set => CheckError(Api.rtGeometryGroupSetAcceleration(InternalPtr, value.InternalPtr));

        }

        public NodeCollection<GeometryInstance> Collection => mCollection;

        public void AddChild(GeometryInstance instance)
        {
            SetChild(ChildCount++, instance);
        }

        public void AddChildren(IEnumerable<GeometryInstance> instances)
        {
            if (instances == null)
                return;

            foreach (var gi in instances)
            {
                SetChild(ChildCount++, gi);
            }
        }

        private GeometryInstance GetChild(int index)
        {
            if (index >= ChildCount)
            {
                throw new ArgumentOutOfRangeException("index");
            }

            CheckError(Api.rtGeometryGroupGetChild(InternalPtr, (uint)index, out var instance));

            return new GeometryInstance(mContext, instance);
        }

        private void SetChild(int i, GeometryInstance instance)
        {
            CheckError(Api.rtGeometryGroupSetChild(InternalPtr, (uint)i, instance.InternalPtr));
        }
    }
}