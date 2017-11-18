using System;

namespace OptixCore.Library
{
    public abstract  class DataNode : OptixNode
    {
        public abstract IntPtr ObjectPtr();

        protected DataNode(Context context) : base(context)
        {
        }
    }
}