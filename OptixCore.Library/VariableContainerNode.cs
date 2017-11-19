using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public abstract class VariableContainerNode : DefaultOptixNode, IVariableContainer
    {
        protected abstract Func<int, IntPtr> GetVariable { get; }
        protected abstract Func<string, IntPtr> QueryVariable { get; }
        protected abstract Func<string, IntPtr> DeclareVariable { get; }
        protected abstract Func<IntPtr, RTresult> RemoveVariable { get; }
        protected abstract Func<int> GetVariableCount { get; }
        public Variable this[int index]
        {
            get
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");
                var rtVar = GetVariable(index);
                return new Variable(mContext, rtVar);

            }
            set
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                var rtVar = GetVariable(index);

                if (value == null)
                {
                    CheckError(RemoveVariable(rtVar));
                }
                else
                {
                    throw new OptixException($"{GetType().Name} Error: Variable copying not yet implemented");
                }
            }
        }

        public Variable this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException($"{GetType().Name} Error: Variable name is null or empty");

                var rtVar = QueryVariable(name);

                if (rtVar == IntPtr.Zero)
                    rtVar = DeclareVariable(name);

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException($"{GetType().Name} Error: Variable name is null or empty");


                var rtVar = QueryVariable(name);


                if (rtVar != IntPtr.Zero && value == null)
                {
                    RemoveVariable(rtVar);
                }
                else
                {
                    if (string.IsNullOrEmpty(name))
                        throw new OptixException($"{GetType().Name} Error: Variable copying not yet implemented");
                }
            }
        }

        public int VariableCount => GetVariableCount();

        protected VariableContainerNode(Context context) : base(context)
        {
        }
    }
}