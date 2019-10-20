using System;
using System.Numerics;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public class Variable : OptixNode
    {
        protected IntPtr mVariable;

        public Variable(Context context, IntPtr var) : base(context)
        {
            mVariable = var;
        }
        public override void Validate()
        {
        }

        public override void Destroy()
        {
        }

        public int GetInt()
        {
            int x = 0;
            CheckError(Api.rtVariableGet1i(mVariable, ref x));
            return x;
        }

        public uint GetUInt()
        {
            var x = 0u;
            CheckError(Api.rtVariableGet1ui(mVariable, ref x));
            return x;
        }

        public float GetFloat()
        {
            float x = 0f;
            CheckError(Api.rtVariableGet1f(mVariable, ref x));
            return x;
        }

        public Vector2 GetFloat2()
        {
            var var = new Vector2();
            CheckError(Api.rtVariableGet2fv(mVariable, ref var));
            return var;
        }

        public Vector3 GetFloat3()
        {
            var var = new Vector3();
            CheckError(Api.rtVariableGet3fv(mVariable, ref var));
            return var;
        }


        public Int3 GetInt3()
        {
            var var = new Int3();
            CheckError(Api.rtVariableGet3iv(mVariable, ref var));
            return var;
        }

        public Vector4 GetFloat4()
        {
            var var = new Vector4();
            CheckError(Api.rtVariableGet4fv(mVariable, ref var));
            return var;
        }

        public Matrix4x4 GetMatrix4x4(bool transpose = false)
        {
            var var = new Matrix4x4();
            CheckError(Api.rtVariableGetMatrix4x4fv(mVariable, transpose ? 1 : 0, ref var));
            return var;
        }

        public void Set(int x)
        {
            CheckError(Api.rtVariableSet1i(mVariable, x));
        }
        public void Set(uint x)
        {
            CheckError(Api.rtVariableSet1ui(mVariable, x));
        }
        public void Set(float x)
        {
            if (float.IsNaN(x) || float.IsInfinity(x))
                Console.WriteLine("Invalid variable value!!!");
            CheckError(Api.rtVariableSet1f(mVariable, x));
        }

        public void Set(ref Vector2 x)
        {
            CheckError(Api.rtVariableSet2f(mVariable, x.X, x.Y));
        }

        public void Set(ref Int3 x)
        {
            CheckError(Api.rtVariableSet3iv(mVariable, ref x));
        }


        public void Set(ref Vector3 x)
        {
            CheckError(Api.rtVariableSet3f(mVariable, x.X, x.Y, x.Z));
        }

        public void Set(ref Vector4 x)
        {
            CheckError(Api.rtVariableSet4f(mVariable, x.X, x.Y, x.Z, x.W));
        }
        public void Set(float x, float y)
        {
            CheckError(Api.rtVariableSet2f(mVariable, x, y));
        }

        public void Set(float x, float y, float z)
        {
            CheckError(Api.rtVariableSet3f(mVariable, x, y, z));
        }

        public void Set(float x, float y, float z, float w)
        {
            CheckError(Api.rtVariableSet4f(mVariable, x, y, z, w));
        }

        public void Set(ref Matrix4x4 m, bool transpose = false)
        {
            CheckError(Api.rtVariableSetMatrix4x4fv(mVariable, transpose ? 1 : 0, m));
        }

        public void Set(DataNode obj)
        {
            CheckError(Api.rtVariableSetObject(mVariable, obj.InternalPtr));
        }

        public void Set(OptixNode obj)
        {
            CheckError(Api.rtVariableSetObject(mVariable, obj.InternalPtr));
        }

        public void SetFloat3(ref Vector3 var)
        {
            CheckError(Api.rtVariableSet3fv(mVariable, var));
        }

        public void SetFloat3(Vector3 var)
        {
            CheckError(Api.rtVariableSet3fv(mVariable, var));
        }

        /// <summary>
        /// Gets the type of object set on the variable
        /// </summary>
        public OptixTypes OptixTypes
        {
            get
            {
                RTobjecttype type = RTobjecttype.RT_OBJECTTYPE_UNKNOWN;
                CheckError(Api.rtVariableGetType(mVariable, ref type));

                return (OptixTypes)type;
            }
        }

        /*
         	generic< typename T > where T : value class
		T GetUserData()
		{
			T var;
			pin_ptr<T> ptr = &var;
			CheckError( rtVariableGetUserData( mVariable, sizeof( T ), ptr ) );

			return var;
		}

		/// <summary>
		/// Sets the user data on the object
        /// </summary>
		generic< typename T > where T : value class
		void SetUserData( T var )
		{
			pin_ptr<T> ptr = &var;
			CheckError( rtVariableSetUserData( mVariable, sizeof( T ), ptr ) );
		}

		/// <summary>
		/// Gets the annotation set on the variable in the cuda program
        /// </summary>
		property System::String^ Annotation
		{
			System::String^ get()
			{
				const char* str = 0;
				CheckError( rtVariableGetAnnotation( mVariable, &str ) );

				return AnsiiToStr( str );
			}
		}

		

         
         
         */
    }
}

