using System;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    /// <summary>
    /// Specifies the direction of data flow from the host to the OptiX devices.
    /// </summary>
    [Flags]
    public enum BufferType
    {
        /// <summary>
        /// Invalid BufferType used for error catching.
        /// </summary>
        None = 0,

        /// <summary>
        /// OptixBuffer will provide data input to OptiX devices. Only the host can write data to the buffer.
        /// </summary>
        Input = RTbuffertype.RT_BUFFER_INPUT,

        /// <summary>
        /// OptixBuffer will provide data output to the host. Only the OptiX device can write data to the buffer.
        /// </summary>
        Output = RTbuffertype.RT_BUFFER_OUTPUT,

        /// <summary>
        /// OptixBuffer will provide data output to the host and data input to OptiX devices. 
        /// Data can be read and written by both host and OptiX device.
        /// </summary>
        InputOutput = RTbuffertype.RT_BUFFER_INPUT_OUTPUT,

        /// <summary>
        /// OptixBuffer will provide the host to write data, and the OptiX device to read and write data. 
        /// The buffer will never be copied back to the host. A seperate copy lives on each OptiX device.<br/>
        /// Useful for accumulating intermediate results on OptiX devices in multi-gpu environments.
        /// </summary>
        InputOutputLocal = RTbuffertype.RT_BUFFER_INPUT_OUTPUT | RTbufferflag.RT_BUFFER_GPU_LOCAL,
    };

    /// <summary>
    /// Description data used to create a <see cref="OptixBuffer">OptixBuffer</see>.
    /// </summary>
    public struct BufferDesc
    {
        /// <summary>
        /// Width of the OptixBuffer.
        /// </summary>
        public ulong Width;

        /// <summary>
        /// Height of the OptixBuffer.
        /// </summary>
        public ulong Height;

        /// <summary>
        /// Depth of the OptixBuffer.
        /// </summary>
        public ulong Depth;

        /// <summary>
        /// Size of the Type the buffer contains. Only needed for <see cref="Format.User">Format.User</see> formats.
        /// </summary>
        public ulong ElemSize;

        /// <summary>
        /// <see cref="BufferType">Type</see> of the OptixBuffer.
        /// </summary>
        public BufferType Type;

        /// <summary>
        /// <see cref="Format">Format</see> of the OptixBuffer
        /// </summary>
        public Format Format;

        public BufferDesc(ulong width, ulong height, ulong depth, ulong elemsize, BufferType type, Format format)
        {
            Width = width;
            Height = height;
            Depth = depth;
            ElemSize = elemsize;
            Type = type;
            Format = format;
        }

        /// <summary>
        /// Gets a default BufferDesc with safe initilaized values.
        /// Width = 1
        /// Height = 1
        /// Depth = 1
        /// ElemSize = 0
        /// Type = None
        /// Format = Unknown
        /// </summary>
        public static BufferDesc Default
        {
            get
            {
                BufferDesc desc = new BufferDesc(1, 1, 1, 0, BufferType.None, Format.Unknown);
                return desc;
            }
        }
    };

    public class OptixBuffer : DataNode
    {
        protected ulong mWidth;
        protected ulong mHeight;
        protected ulong mDepth;
        protected Format mFormat;
        protected BufferType mType;
        public OptixBuffer(Context context, BufferDesc desc) : base(context)
        {
            Create(desc);
        }

        public OptixBuffer(Context c) : base(c)
        {

        }

        public void Create(BufferDesc desc)
        {
            mFormat = desc.Format;
            mType = desc.Type;
            mWidth = desc.Width;
            mHeight = desc.Height;
            mDepth = desc.Depth;

            if (mFormat == Format.User && desc.ElemSize <= 0)
                throw new Exception("Buffer Error: Invalid ElemSize for User format Buffer.");


            CheckError(Api.rtBufferCreate(mContext.InternalPtr, (uint)mType, ref InternalPtr));
            CheckError(Api.rtBufferSetFormat(InternalPtr, (RTformat)mFormat));
            SetSize(mWidth, mHeight, mDepth);

            if (mFormat == Format.User)
                CheckError(Api.rtBufferSetElementSize(InternalPtr, (uint)desc.ElemSize));
        }

        private void SetSize(ulong width, ulong height, ulong depth)
        {
            if (height <= 1 && mDepth <= 1)
                CheckError(Api.rtBufferSetSize1D(InternalPtr, (uint)width));
            else if (depth <= 1)
                CheckError(Api.rtBufferSetSize2D(InternalPtr, (uint)width, (uint)height));
            else
                CheckError(Api.rtBufferSetSize3D(InternalPtr, (uint)width, (uint)height, (uint)depth));
        }

        public override void Validate()
        {
            CheckError(Api.rtBufferValidate(InternalPtr));
        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtBufferDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
        }

        public void SetData<T>(T[] data)
            where T : struct
        {
            if ((mType & BufferType.Input) != BufferType.Input)
                throw new OptixException("OptixBuffer Error: Attempting to set data on a non Input buffer!");

            if (data == null || data.Length == 0)
                return;

            uint mySize = 0;
            CheckError(Api.rtBufferGetElementSize(InternalPtr, ref mySize));

            mySize *= (uint)mWidth;
            if (mHeight > 0)
                mySize *= (uint)mHeight;
            if (mDepth > 0)
                mySize *= (uint)mDepth;

            uint size = (uint)(data.Length * Marshal.SizeOf<T>());

            if (size > mySize)
                throw new OptixException("OptixBuffer Error: Attempting to set data that is larger than the buffer size");

            var l = GCHandle.Alloc(data, GCHandleType.Pinned);
            var bufferMap = IntPtr.Zero;
            CheckError(Api.rtBufferMap(InternalPtr, ref bufferMap));
            if (bufferMap == IntPtr.Zero)
            {
                throw new OptixException("Buffer Error: Internal buffer cannot be mapped");
            }
            MemoryHelper.CopyMemory(bufferMap, l.AddrOfPinnedObject(), size);
            CheckError(Api.rtBufferUnmap(InternalPtr));
            l.Free();
        }

        public void GetData<T>(ref T[] data)
            where T : struct
        {
            if ((mType & BufferType.Output) != BufferType.Output)
                throw new OptixException("Buffer Error: Attempting to get data on a non Output buffer!");

            uint size = 0;
            var tSize = Marshal.SizeOf<T>();

            var length = mWidth;
            if (mHeight > 0)
                length *= mHeight;
            if (mDepth > 0)
                length *= mDepth;

            CheckError(Api.rtBufferGetElementSize(InternalPtr, ref size));

            if (tSize != size)
                throw new OptixException("Buffer Error: Array Type is not of equal size to Buffer Format type.");

            if (data == null)
                data = new T[length * (ulong)tSize];

            size *= (uint)length;

            var l = GCHandle.Alloc(data, GCHandleType.Pinned);
            var bufferMap = IntPtr.Zero;
            CheckError(Api.rtBufferMap(InternalPtr, ref bufferMap));
            if (bufferMap == IntPtr.Zero)
            {
                throw new OptixException("Buffer Error: Internal buffer cannot be mapped");
            }
            MemoryHelper.CopyMemory(l.AddrOfPinnedObject(), bufferMap, size);
            l.Free();
            CheckError(Api.rtBufferUnmap(InternalPtr));
        }


        public void GetDataNoAlloc<T>(T[] data)
            where T : struct
        {
            if ((mType & BufferType.Output) != BufferType.Output)
                throw new OptixException("Buffer Error: Attempting to get data on a non Output buffer!");

            uint size = 0;
            var tSize = Marshal.SizeOf<T>();

            var length = mWidth;
            if (mHeight > 0)
                length *= mHeight;
            if (mDepth > 0)
                length *= mDepth;

            CheckError(Api.rtBufferGetElementSize(InternalPtr, ref size));

            if (tSize != size)
                throw new OptixException("Buffer Error: Array Type is not of equal size to Buffer Format type.");

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            size *= (uint)length;

            var l = GCHandle.Alloc(data, GCHandleType.Pinned);
            var bufferMap = IntPtr.Zero;
            CheckError(Api.rtBufferMap(InternalPtr, ref bufferMap));
            if (bufferMap == IntPtr.Zero)
            {
                throw new OptixException("Buffer Error: Internal buffer cannot be mapped");
            }
            MemoryHelper.CopyMemory(l.AddrOfPinnedObject(), bufferMap, size);
            l.Free();
            CheckError(Api.rtBufferUnmap(InternalPtr));
        }

        public BufferStream Map()
        {
            uint size = 0;
            CheckError(Api.rtBufferGetElementSize(InternalPtr, ref size));

            var bufferMap = IntPtr.Zero;
            CheckError(Api.rtBufferMap(InternalPtr, ref bufferMap));
            if (bufferMap == IntPtr.Zero)
            {
                throw new OptixException("Buffer Error: Internal buffer cannot be mapped");
            }

            bool canRead = (mType & BufferType.Output) == BufferType.Output;
            bool canWrite = (mType & BufferType.Input) == BufferType.Input;

            size *= (uint)mWidth;
            if (mHeight > 0)
                size *= (uint)mHeight;
            if (mDepth > 0)
                size *= (uint)mDepth;

            return new BufferStream(bufferMap, size, canRead, canWrite, false);
        }

        public void Unmap()
        {
            CheckError(Api.rtBufferUnmap(InternalPtr));
        }

        public BufferType Type => mType;

        public Format Format => mFormat;

        public ulong Width => mWidth;
        public ulong Height => mHeight;
        public ulong Depth => mDepth;

        public ulong ElemSize
        {
            get
            {
                var size = 0u;
                CheckError(Api.rtBufferGetElementSize(InternalPtr, ref size));
                return size;
            }
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }
    }
}