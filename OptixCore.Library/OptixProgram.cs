using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class OptixProgram : OptixNode, IVariableContainer
    {
        public OptixProgram(Context context, string fileName, string programName) : base(context)
        {
            if (string.IsNullOrWhiteSpace(fileName) || string.IsNullOrWhiteSpace(programName))
            {
                throw new OptixException("Program Error: Null or Empty filename or program name");
            }
            CheckError(Api.rtProgramCreateFromPTXFile(context.InternalPtr, fileName, programName,out InternalPtr));
        }

        public OptixProgram(Context context, IntPtr program) : base(context)
        {
            InternalPtr = program;
        }

        public void SetProgram(string name, OptixProgram @object)
        {
            var v = this[name].InternalPtr;
            var pr = @object.InternalPtr;
            CheckError(Api.rtVariableSetObject(v, pr));
        }


        public Variable this[int index]
        {
            get
            {
                if (index < 0 || index >= VariableCount)
                    throw new ArgumentOutOfRangeException("index");

                var rtVar = IntPtr.Zero;

                CheckError(Api.rtProgramGetVariable(InternalPtr, (uint)index, rtVar));

                return new Variable(mContext, rtVar);
            }
            set { throw new System.NotImplementedException(); }
        }

        public Variable this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Program Error: Variable name is null or empty");


                var rtVar = InternalPtr;
                CheckError(Api.rtProgramQueryVariable(InternalPtr, name, rtVar));

                if (rtVar == IntPtr.Zero)
                    CheckError(Api.rtProgramDeclareVariable(InternalPtr, name, rtVar));

                return new Variable(mContext, rtVar);
            }
            set
            {
                if (string.IsNullOrEmpty(name))
                    throw new OptixException("Program Error: Variable name is null or empty");

                var rtVar = IntPtr.Zero;

                CheckError(Api.rtProgramQueryVariable(InternalPtr, name, rtVar));

                if (rtVar != IntPtr.Zero && value == null)
                {
                    CheckError(Api.rtProgramRemoveVariable(InternalPtr, rtVar));
                }
                else
                {
                    throw new OptixException("Program Error: Variable copying not yet implemented");
                }
            }
        }

        /*	

	void Program::default::set( int index, Variable^ var )
	{
		if( index < 0 || index >= VariableCount )
			throw gcnew ArgumentOutOfRangeException( "index" );

		RTvariable rtVar;
		CheckError( rtProgramGetVariable( mProgram, index, &rtVar ) );

		if( var == nullptr )
		{
			CheckError( rtProgramRemoveVariable( mProgram, rtVar ) );
		}
		else
		{
			throw gcnew Exception( "Program Error: Variable copying not yet implemented" );
		}
	}
        
              
             */
        public override void Validate()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtProgramValidate(InternalPtr));
        }

        public override void Destroy()
        {
            CheckError(Api.rtProgramDestroy(InternalPtr));
        }

        public int VariableCount
        {
            get
            {
                var count = 0u;
                CheckError(Api.rtProgramGetVariableCount(InternalPtr, ref count));
                return (int)count;
            }
        }

    }
}