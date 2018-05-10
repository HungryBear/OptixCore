using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library
{
    public class TraversalStream<T>
        where T : struct
    {
        public BufferStream Stream;

        public long Length => Stream.Length / Marshal.SizeOf<T>();

        public TraversalStream(BufferStream source)
        {
            this.Stream = source;
        }

        public void GetData(T[] results)
        {
            if (results == null)
                throw new ArgumentNullException("results", "Results array cannot be null.");

            if ((results.Length * Marshal.SizeOf<T>()) != Stream.Length)
                throw new ArgumentOutOfRangeException("results", "Results array must be able to hold entire TraversalStream");

            Stream.ReadRange(results, 0, results.Length);
        }
    }
}