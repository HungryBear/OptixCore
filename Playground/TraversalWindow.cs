using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Numerics;
using OpenTK.Graphics.OpenGL;
using OptixCore.Library;
using OptixCore.Library.Native.Prime;
using OptixCore.Library.Scene;


namespace Playground
{
    public class TraversalWindow : OptixWindow
    {
        TraversalEngine mTraversal;

        float[] mDepths;
        Vector3[] mNormals;
        bool mUpdateDepth = true;
        bool mUpdateNormals = true;

        public TraversalWindow() : base(800, 600)
        {
        }

        protected override void Initialize()
        {
            base.Initialize();
            var modelName = "sibenik.obj";
            var modelPath = Path.GetFullPath(@"..\..\..\..\..\Assets\Models\" + modelName);
            OptixContext = new Context();
            mTraversal = new TraversalEngine(OptixContext, QueryType.ClosestHit, RayFormat.OriginDirectionInterleaved, TriFormat.Mesh,
                TraversalOutput.Normal, InitOptions.None);
            mTraversal.SetCpuThreadCount(1);
            //mTraversal.NumCpuThreads = Context.CpuNumThreads;


            var model = new OptixOBJLoader(modelPath, OptixContext, null, null);

            //OBJLoader normally automatically creates Geometry, GeometryInstances, and GeometryGroups for Optix
            //but the Traversal API doesn't use that, so turn that off
            model.ParseMaterials = false;
            model.ParseNormals = false;
            model.GenerateNormals = false;
            model.GenerateGeometry = false;
            model.LoadContent();
            SetCamera(model.BBox);

            //we copy the data here or else the GC would clean up a model.Vertices.ToArray() for example
            var verts = new Vector3[model.Positions.Count];
            var tris = new Int3[model.Groups[0].VIndices.Count];

            model.Positions.CopyTo(verts);
            model.Groups[0].VIndices.CopyTo(tris);
            var indexes = tris.SelectMany(c => new[] { c.X, c.Y, c.Z }).ToArray();
            mTraversal.SetMesh(verts, indexes);

            int numRays = Width * Height;
            var rays = CreateRays();
            mTraversal.SetRayData<TraversalEngine.Ray>(rays);

            mTraversal.Preprocess();
            mDepths = new float[numRays];
            mNormals = new Vector3[numRays];

            RaysTracedPerFrame = numRays;

            Console.WriteLine("Accel data size " + mTraversal.AccelDataSize);
            using (var accData = mTraversal.AccelData)
            {
                var size = accData.Length;
                var buffer = new byte[size];
                accData.ReadRange(buffer, 0, (int)size);
                var nonZeros = buffer.Count(b => b != 0);
                Console.WriteLine("Non zeros in acc data " + nonZeros);
            }

        }

        protected override void RayTrace()
        {
            mTraversal.Traverse();
        }

        protected override void Display()
        {
            //base.Display();
            /*
            if (mUpdateDepth)
            {
                float maxDepth = -1.0f;
                var stream = mTraversal.MapResults();
                var allData = new TraversalResult[Width * Height];
                //mTraversal.GetResults(allData);
                stream.Stream.ReadRange(allData, 0, (int)stream.Length);
                int nonzero = allData.Count(d => d.T > 0);

                //stream.GetData(allData);
                //mDepths = allData.Select(c => c.T).ToArray();
                //maxDepth = allData.Max(c => c.T);

                for (int i = 0; i < stream.Length; i++)
                {
                    //var result = stream.Stream.Read<TraversalResult>();
                    mDepths[i] = allData[i].T;

                    maxDepth = Math.Max(maxDepth, mDepths[i]);
                }
                
                if (Math.Abs(maxDepth) < 1e-10f)
                {
                    maxDepth = 1;
                }

                for (int i = 0; i < Width * Height; i++)
                {
                    mDepths[i] /= maxDepth;
                }
                mTraversal.UnmapResults();

                mUpdateDepth = false;
                if (maxDepth > 1)
                {
                    Console.WriteLine("maxdepth = " + maxDepth);
                }
            }

            GL.DrawPixels(Width, Height, PixelFormat.Luminance, PixelType.Float, mDepths);
            */

            if (mUpdateNormals)
            {
                mTraversal.GetNormalOutput(mNormals);

                //bias/scale normals so we don't have black
                for (int i = 0; i < mNormals.Length; i++)
                {
                    mNormals[i] = mNormals[i] * new Vector3(.5f) + new Vector3(.5f);
                }

                mUpdateNormals = false;
            }
            GL.DrawPixels(Width, Height, PixelFormat.Rgb, PixelType.Float, mNormals);

            this.SwapBuffers();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            mTraversal.Dispose();
            base.OnClosing(e);
        }

        private TraversalEngine.Ray[] CreateRays()
        {
            TraversalEngine.Ray[] rays = new TraversalEngine.Ray[Width * Height];

            for (int x = 0; x < Width; x++)
            {
                for (int y = 0; y < Height; y++)
                {
                    Vector2 d = new Vector2(x, y) / new Vector2(Width, Height) * 2.0f - new Vector2(1.0f);

                    TraversalEngine.Ray ray;
                    ray.Origin = Camera.Position;
                    ray.Direction = (d.X * Camera.Right + d.Y * Camera.Up + Camera.Look).NormalizedCopy();

                    rays[y * Width + x] = ray;
                }
            }

            return rays;
        }

        private void SetCamera(BoundingBox box)
        {
            Camera = new Camera();
            Camera.Aspect = (float)Width / (float)Height;
            Camera.Fov = 30;
            Camera.RotationVel = 100.0f;
            Camera.TranslationVel = 500.0f;

            //sibenik camera position
            Camera.LookAt(new Vector3(-19.5f, -10.3f, .8f), new Vector3(0.0f, -13.3f, .8f), Vector3.UnitY);
            //Camera.CenterOnBoundingBox( box );
        }
    }
}