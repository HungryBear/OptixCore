using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Group : ContainerNode, IContainerNode<ContainerNode>, INodeCollectionProvider<ContainerNode>
    {
        NodeCollection<ContainerNode> mCollection;

        public Group(Context context) : base(context)
        {
            mCollection = new NodeCollection<ContainerNode>(this);
            CheckError(Api.rtGroupCreate(context.InternalPtr, ref InternalPtr));
        }

        internal Group(Context context, IntPtr ptr) : base(context)
        {
            mCollection = new NodeCollection<ContainerNode>(this);
            InternalPtr = ptr;
        }

        public override void Validate()
        {
            CheckError(Api.rtGroupValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
            {
                CheckError(Api.rtGroupDestroy(InternalPtr));
                InternalPtr = IntPtr.Zero;
            }
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }

        public ContainerNode this[int index]
        {
            get => mCollection[index];
            set => mCollection[index] = value;
        }

        public int ChildCount { get => mCollection.Count; set => mCollection.Count = value; }
        public NodeCollection<ContainerNode> Collection => mCollection;
    }
}