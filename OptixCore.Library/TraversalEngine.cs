using System;
using System.Numerics;
using System.Runtime.InteropServices;
using OptixCore.Library.Native;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library
{
    public class TraversalEngine : IDisposable
    {
        public struct Ray
        {
            public Vector3 Origin, Direction;
        }

        public struct RayMinMax
        {
            public Vector3 Origin, Direction;
            public float MinT, MaxT;
        }

        IntPtr mTraversal;
        RayFormat mRayFormat;
        int mRaySize;
        int mCurrentNumRays;

        public uint AccelDataSize
        {
            get
            {
                uint size = 0;
                CheckError(TraversalApi.rtuTraversalGetAccelDataSize(mTraversal, ref size));
                return size;
            }
        }

        public BufferStream AccelData
        {
            get
            {
                var size = AccelDataSize;
                var data = Marshal.AllocHGlobal((int)size);
                CheckError(TraversalApi.rtuTraversalGetAccelData(mTraversal, data));
                return new BufferStream(data, size, true, false, true);
            }
        }

        public TraversalEngine(Context context, QueryType query, RayFormat rayFormat, TriFormat triFormat,
                    TraversalOutput outputs, InitOptions options)
        {
            mRayFormat = rayFormat;
            mRaySize = rayFormat == RayFormat.OriginDirectionInterleaved ? 24 : 32;
            CheckError(TraversalApi.rtuTraversalCreate(ref mTraversal, (RTUquerytype)query, (RTUrayformat)rayFormat,
                (RTUtriformat)triFormat, (uint)outputs, (uint)options, context.InternalPtr));
        }

        protected void CheckError(RTresult result)
        {
            if (result != RTresult.RT_SUCCESS)
            {
                TraversalApi.rtuTraversalGetErrorString(mTraversal, result, out var message);
                throw new OptixException($"Optix traversal error : {message} {GetType().Name}");
            }
        }

        public void SetMesh(Vector3[] vertices, int[] indices)
        {
            var vertHandle = GCHandle.Alloc(vertices, GCHandleType.Pinned);
            var indHandle = GCHandle.Alloc(indices, GCHandleType.Pinned);
            CheckError(TraversalApi.rtuTraversalSetMesh(mTraversal, (uint)vertices.Length, vertHandle.AddrOfPinnedObject(),
                (uint)indices.Length / 3, indHandle.AddrOfPinnedObject()));
            vertHandle.Free();
            indHandle.Free();
        }

        public TraversalStream<T> MapRays<T>(int numRays)
                where T : struct
        {

            mCurrentNumRays = numRays;

            var rays = IntPtr.Zero;
            CheckError(TraversalApi.rtuTraversalMapRays(mTraversal, (uint)numRays, ref rays));

            var stream = new BufferStream(rays, mRaySize * numRays, true, true, false);
            return new TraversalStream<T>(stream);
        }

        public void UnmapRays()
        {
            CheckError(TraversalApi.rtuTraversalUnmapRays(mTraversal));
        }

        public void SetRayData(Ray[] rays)
        {
            SetRayData<Ray>(rays);
        }

        public void SetRayData(RayMinMax[] rays)
        {
            SetRayData<RayMinMax>(rays);
        }

        public void SetRayData<T>(T[] rays)
            where T : struct
        {
            if (rays == null || rays.Length == 0)
                return;

            var rayData = IntPtr.Zero;
            var rayPtr = GCHandle.Alloc(rays, GCHandleType.Pinned);

            CheckError(TraversalApi.rtuTraversalMapRays(mTraversal, (uint)rays.Length, ref rayData));

            if (rayData == IntPtr.Zero)
                throw new OptixException("Traversal Error: Ray buffer cannot be mapped");
            var raySpan = new Span<T>(rays);
            MemoryHelper.CopyFromManaged(ref raySpan, rayData, (uint)rays.Length);
            //Unsafe.Copy(rayData.ToPointer(), ref rays);

            CheckError(TraversalApi.rtuTraversalUnmapRays(mTraversal));
            mCurrentNumRays = rays.Length;
            rayPtr.Free();
        }

        public TraversalStream<TraversalResult> MapResults()
        {
            IntPtr results = IntPtr.Zero;

            CheckError(TraversalApi.rtuTraversalMapResults(mTraversal, ref results));

            if (results == IntPtr.Zero)
                throw new OptixException("Traversal Error: Results buffer cannot be mapped");

            var stream = new BufferStream(results, 8 * mCurrentNumRays, true, false, false);
            return new TraversalStream<TraversalResult>(stream);
        }

        public void UnmapResults()
        {
            CheckError(TraversalApi.rtuTraversalUnmapResults(mTraversal));
        }

        public void SetCpuThreadCount(int threadsCount)
        {
            CheckError(TraversalApi.rtuTraversalSetOption(mTraversal, Native.Prime.RunTimeOptions.NumThreads, MemoryHelper.AddressOf(threadsCount)));
        }

        public void Traverse()
        {
            CheckError(TraversalApi.rtuTraversalTraverse(mTraversal));
        }

        public void Dispose()
        {
            if (mTraversal != IntPtr.Zero)
            {
                TraversalApi.rtuTraversalDestroy(mTraversal);
                mTraversal = IntPtr.Zero;
            }
        }

        public void Preprocess()
        {
            CheckError(TraversalApi.rtuTraversalPreprocess(mTraversal));

        }

        public void GetResults(TraversalResult[] results)
        {
            if (results == null || results.Length == 0)
                return;

            var resultData = IntPtr.Zero;
            var resultsPtr = GCHandle.Alloc(results, GCHandleType.Pinned);


            CheckError(TraversalApi.rtuTraversalMapResults(mTraversal, ref resultData));

            if (resultData == IntPtr.Zero)
                throw new OptixException("Traversal Error: Results buffer cannot be mapped");

            MemoryHelper.BlitMemory(resultsPtr.AddrOfPinnedObject(), resultData, (uint)(mCurrentNumRays * 8u));

            CheckError(TraversalApi.rtuTraversalUnmapResults(mTraversal));
            resultsPtr.Free();
        }

        public void GetNormalOutput(Vector3[] output)
        {
            if (output.Length != mCurrentNumRays)
                throw new ArgumentOutOfRangeException("output", "Output array must equal number of rays set on the Traversal object");

            IntPtr outputData = IntPtr.Zero;
            CheckError(TraversalApi.rtuTraversalMapOutput(mTraversal, RTUoutput.RTU_OUTPUT_NORMAL, ref outputData));

            if (outputData == IntPtr.Zero)
                throw new OptixException("Traversal Error: NormalOutput buffer cannot be mapped");
            var lc = GCHandle.Alloc(output, GCHandleType.Pinned);

            var span = new Span<Vector3>(output);
            MemoryHelper.CopyFromUnmanaged(outputData, ref span, (uint)mCurrentNumRays);

            CheckError(TraversalApi.rtuTraversalUnmapOutput(mTraversal, RTUoutput.RTU_OUTPUT_NORMAL));
            lc.Free();
        }

        public void GetBaryCentricOutput(Vector2[] output )
        {
            if (output.Length != mCurrentNumRays)
                throw new ArgumentOutOfRangeException("output", "Output array must equal number of rays set on the Traversal object");

            IntPtr outputData = IntPtr.Zero;
            CheckError(TraversalApi.rtuTraversalMapOutput(mTraversal, RTUoutput.RTU_OUTPUT_BARYCENTRIC, ref outputData));

            if (outputData == IntPtr.Zero)
                throw new OptixException("Traversal Error: NormalOutput buffer cannot be mapped");
            var lc = GCHandle.Alloc(output, GCHandleType.Pinned);

            var span = new Span<Vector2>(output);
            MemoryHelper.CopyFromUnmanaged(outputData, ref span, (uint)mCurrentNumRays);

            CheckError(TraversalApi.rtuTraversalUnmapOutput(mTraversal, RTUoutput.RTU_OUTPUT_BARYCENTRIC));
            lc.Free();
        }
    }
}