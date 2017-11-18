using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public abstract class ContainerNode : DataNode
    {
        public ContainerNode ConstructContainerNodeFromType( OptixTypes optixTypes, IntPtr @object)
        {
            switch (optixTypes)
            {
                case OptixTypes.Group:
                    return new Group(mContext, @object);
                case OptixTypes.GeometryGroup:
                    return new GeometryGroup(mContext, @object);
                //case OptixTypes.Selector:
                //    return new Selector(mContext, static_cast<RTselector>(object));
                //    break;
                case OptixTypes.Transform:
                    return new Transform(mContext, @object);
                default:
                    throw new Exception("Internal Error: Unrecognized type!");
            }
        }

        protected ContainerNode(Context context) : base(context)
        {
        }
    }
}