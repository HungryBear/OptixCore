﻿using System;
using System.IO;
using System.Runtime.InteropServices;

namespace OptixCore.Library
{
    public class BufferStream : Stream
    {
        protected IntPtr Buffer;
        protected bool mCanRead;
        protected bool mCanWrite;
        protected bool mOwnData;

        protected long mSize;
        protected long mPosition;

        public IntPtr DataPointer => Buffer;

        public BufferStream(Stream source)
        {
            if (!source.CanRead)
                throw new NotSupportedException("Optix Error: BufferStream: source Stream does not support reading.");

            mPosition = 0;
            mSize = source.Length;

            Buffer = Marshal.AllocHGlobal((int)mSize);

            mCanRead = true;
            mCanWrite = true;
            mOwnData = true;

            if (mOwnData)
            {
                GC.AddMemoryPressure(mSize);
            }
            var buffer = new byte[4096];

            int num = 0;
            while ((num = source.Read(buffer, 0, buffer.Length)) != 0)
            {
                Write(buffer, 0, num);
            }

            source.Close();
            mCanWrite = false;
        }

        internal BufferStream(IntPtr buffer, Int64 sizeInBytes, bool canRead, bool canWrite, bool ownData)
        {
            Buffer = buffer;
            mSize = sizeInBytes;

            mPosition = 0;

            mCanRead = canRead;
            mCanWrite = canWrite;
            mOwnData = ownData;

            if (ownData)
            {
                GC.AddMemoryPressure(sizeInBytes);
            }
        }

        public override void Close()
        {
            if (Buffer == IntPtr.Zero) return;
            Marshal.FreeHGlobal(Buffer);
            GC.RemoveMemoryPressure(mSize);
            Buffer = IntPtr.Zero;
        }

        public T Read<T>()
            where T : struct
        {
            var t = default(T);
            Read(ref t);
            return t;
        }

        public void Read<T>(ref T r)
            where T : struct
        {
            if (!mCanRead)
                throw new NotSupportedException("Optix Error: BufferStream: Cannot read from non Output Optix buffers");

            var elemSize = Marshal.SizeOf<T>();
            var sizeInBytes = elemSize;

            // internal checks
            if (mPosition + sizeInBytes > mSize)
                throw new EndOfStreamException();

            var l = GCHandle.Alloc(r, GCHandleType.Pinned);
            var dest = new[] { MemoryHelper.AddressOf(r) };
            Marshal.Copy(Buffer, dest, (int)mPosition, sizeInBytes);
            mPosition += sizeInBytes;
            l.Free();
        }

        public int ReadRange<T>(T[] buffer, int offset, int numElems)
            where T : struct
        {
            if (!mCanRead)
                throw new NotSupportedException("Optix Error: BufferStream: Cannot read from non Output Optix buffers");

            var elemSize = Marshal.SizeOf<T>();
            var sizeInBytes = numElems * elemSize;

            //perform some validation steps
            if (buffer == null)
                throw new ArgumentNullException("buffer");

            if (offset < 0)
                throw new ArgumentOutOfRangeException("offset");

            if (sizeInBytes < 0 || (offset + sizeInBytes > buffer.Length))
                throw new ArgumentOutOfRangeException("numElems");

            var validSize = Math.Min(mSize - mPosition, sizeInBytes);

            var l = GCHandle.Alloc(buffer[offset], GCHandleType.Pinned);
            var dest = new[] { l.AddrOfPinnedObject() };
            Marshal.Copy(Buffer, dest, (int)mPosition, (int)validSize);
            l.Free();
            mPosition += validSize;

            return (int)validSize;
        }


        public void Write<T>(T r)
            where T : struct
        {
            if (!mCanWrite)
                throw new NotSupportedException("Optix Error: BufferStream: Cannot write to non Input Optix buffers");

            var elemSize = Marshal.SizeOf<T>();
            var sizeInBytes = elemSize;

            // internal checks
            if (mPosition + sizeInBytes > mSize)
                //throw new EndOfStreamException();
                return;
            var l = GCHandle.Alloc(r, GCHandleType.Pinned);
            var rpt = IntPtr.Zero;
            MemoryHelper.MemCopy(IntPtr.Add(Buffer, (int)mPosition), l.AddrOfPinnedObject(), (uint)sizeInBytes);
            //Marshal.Copy(l.AddrOfPinnedObject(), new[] { Buffer }, (int)mPosition, sizeInBytes);
            mPosition += sizeInBytes;
            l.Free();
        }
        public void WriteRange<T>(T[] buffer, int offset, int numElems)
            where T : struct
        {
            if (!mCanWrite)
                throw new NotSupportedException("Optix Error: BufferStream: Cannot write to non Input Optix buffers");
            var elemSize = Marshal.SizeOf<T>();
            var sizeInBytes = numElems * elemSize;

            //perform some validation steps
            if (buffer == null)
                throw new ArgumentNullException("buffer");

            if (offset < 0)
                throw new ArgumentOutOfRangeException("offset");

            if (sizeInBytes < 0 || (offset + sizeInBytes > buffer.Length))
                throw new ArgumentOutOfRangeException("numElems");
            if (mPosition + sizeInBytes > mSize)
                throw new EndOfStreamException();
            var l = GCHandle.Alloc(buffer[offset], GCHandleType.Pinned);
            Marshal.Copy(l.AddrOfPinnedObject(), new[] { Buffer }, (int)mPosition, sizeInBytes);
            l.Free();
            mPosition += sizeInBytes;
        }

        public bool Save(string path)
        {
            if (!mCanRead)
                throw new NotSupportedException("Optix Error: BufferStream: does not support reading.");

            var buffer = new byte[4096];
            using (var stream = File.OpenWrite(path))
            {

                var num = 0;
                while ((num = Read(buffer, 0, buffer.Length)) != 0)
                {
                    stream.Write(buffer, 0, num);
                }
            }
            return true;
        }

        public override void Flush()
        {
            throw new NotImplementedException();
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            return ReadRange(buffer, offset, count);
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long pos = 0;

            switch (origin)
            {
                case SeekOrigin.Begin:
                    pos = offset;
                    break;

                case SeekOrigin.Current:
                    pos = mPosition + offset;
                    break;

                case SeekOrigin.End:
                    pos = mSize - offset;
                    break;
            }

            if (pos < 0)
                throw new InvalidOperationException("Cannot seek beyond the beginning of the stream.");
            if (pos > mSize)
                throw new InvalidOperationException("Cannot seek beyond the end of the stream.");

            mPosition = pos;
            return mPosition;
        }

        public override void SetLength(long value)
        {
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            WriteRange(buffer, offset, count);
        }

        public override bool CanRead { get => mCanRead; }
        public override bool CanSeek => true;
        public override bool CanWrite { get => mCanWrite; }
        public override long Length { get => mSize; }
        public override long Position { get => mPosition; set => mPosition = value; }
    }


}