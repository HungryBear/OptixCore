using System;
using System.Runtime.Serialization;

namespace OptixCore.Library
{
    public class OptixException : Exception
    {
        public OptixException()
        {
        }

        public OptixException(string message) : base(message)
        {
        }

        public OptixException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected OptixException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }

        public override string ToString()
        {
            return $"{Message} {StackTrace}";
        }
    }
}