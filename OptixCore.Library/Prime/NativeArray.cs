using System;
using System.Runtime.InteropServices;

namespace OptixCore.Library.Prime
{
    internal class NativeArray<T> 
        where T : struct
    {
        readonly IntPtr arrayPtr;

        int sizeofT => Marshal.SizeOf(typeof(T));

        private readonly int _length;
        public T this[int i]
        {
            get => (T)Marshal.PtrToStructure(arrayPtr + i * sizeofT, typeof(T));
            set => Marshal.StructureToPtr(value, arrayPtr + i * sizeofT, false);
        }
        public NativeArray(int length)
        {
            arrayPtr = Marshal.AllocHGlobal(sizeofT * length);
            _length = length;
            GC.AddMemoryPressure(sizeofT * length);
        }
        public NativeArray(T[] elements)
        {
            arrayPtr = Marshal.AllocHGlobal(sizeofT * elements.Length);
            _length = elements.Length;

            GC.AddMemoryPressure(sizeofT * elements.Length);
            var span = new Span<T>(elements);
            MemoryHelper.CopyFromManaged(ref span, arrayPtr, (uint)elements.Length);
        }
        ~NativeArray()
        {
            GC.RemoveMemoryPressure(sizeofT * _length);
            Marshal.FreeHGlobal(arrayPtr);
        }
    }
}