using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public abstract class DefaultOptixNode : OptixNode, IVariableContainer
    {
        protected abstract Func<RTresult> ValidateAction { get; }
        protected abstract Func<RTresult> DestroyAction { get; }
        protected abstract Func<int> GetVariableCount { get; }

        protected abstract Func<int, IntPtr> GetVariable { get; }
        protected abstract Func<string, IntPtr> QueryVariable { get; }
        protected abstract Func<string, IntPtr> DeclareVariable { get; }

        protected abstract Action<int, IntPtr> SetVariable { get; }
        protected abstract Func<IntPtr, RTresult> RemoveVariable { get; }


        protected DefaultOptixNode(Context context) : base(context)
        {
        }


        public override void Validate()
        {
            CheckError(ValidateAction());
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(DestroyAction());
            InternalPtr = IntPtr.Zero;
        }


        Variable IVariableContainer.this[int index]
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

        Variable IVariableContainer.this[string name]
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
    }
}