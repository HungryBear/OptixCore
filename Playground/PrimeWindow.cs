using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Numerics;
using OpenTK.Graphics.OpenGL;
using OptixCore.Library;
using OptixCore.Library.Prime;
using OptixCore.Library.Scene;
using RayFormat = OptixCore.Library.Prime.RayFormat;


namespace Playground
{
    public class PrimeWindow : OptixWindow
    {
        PrimeEngine mTraversal;

        float[] mDepths;
        PrimeEngine.Hit[] hits;
        PrimeEngine.Ray[] rays;

        Vector3[] mNormals;
        bool mUpdateDepth = true;
        bool mUpdateNormals = true;

        public PrimeWindow() : base(800, 600)
        {
        }

        protected override void Initialize()
        {
            base.Initialize();
            var modelName = "sibenik.obj";
            var modelPath = Path.GetFullPath(@"..\..\..\..\..\Assets\Models\" + modelName);
            mTraversal = new PrimeEngine(RayFormat.OriginDirectionMinMaxInterleaved, RayHitType.Closest);

            var model = new OBJLoader(modelPath);

            model.ParseMaterials = false;
            model.ParseNormals = false;
            model.GenerateNormals = false;
            model.GenerateGeometry = false;
            model.LoadContent();
            SetCamera(model.BBox);

            var verts = new Vector3[model.Positions.Count];
            var tris = new Int3[model.Groups[0].VIndices.Count];

            model.Positions.CopyTo(verts);
            model.Groups[0].VIndices.CopyTo(tris);
            var indexes = tris.SelectMany(c => new[] { c.X, c.Y, c.Z }).ToArray();
            mTraversal.SetMesh(verts, indexes);

            int numRays = Width * Height;
            var rays = CreateRays();
            mTraversal.SetRays(rays);

            mDepths = new float[numRays];
            mNormals = new Vector3[numRays];

            RaysTracedPerFrame = numRays;

        }

        protected override void RayTrace()
        {
            hits = mTraversal.Query();
        }

        protected override void Display()
        {
            if (mUpdateNormals && hits != null)
            {
                mNormals = new Vector3[hits.Length];

                //bias/scale normals so we don't have black
                for (int i = 0; i < mNormals.Length; i++)
                {
                    if (hits[i].t > 1e-4f)
                        mNormals[i] = (hits[i].t/10f) * new Vector3(.5f,hits[i].u,0) + new Vector3(.5f,0, hits[i].v);
                    else
                        mNormals[i] = new Vector3(.5f);
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

        private PrimeEngine.Ray[] CreateRays()
        {
            rays = new PrimeEngine.Ray[Width * Height];

            for (int x = 0; x < Width; x++)
            {
                for (int y = 0; y < Height; y++)
                {
                    Vector2 d = new Vector2(x, y) / new Vector2(Width, Height) * 2.0f - new Vector2(1.0f);

                    PrimeEngine.Ray ray = new PrimeEngine.Ray
                    {
                        origin = Camera.Position,
                        tmin = 1e-4f,
                        dir = (d.X * Camera.Right + d.Y * Camera.Up + Camera.Look).NormalizedCopy(),
                        tmax = 1e34f
                    };

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

        protected override void CameraUpdate()
        {
            base.CameraUpdate();
            Console.WriteLine($"Camera = {Camera.Position}f  {Camera.Target}f  {Camera.Up}f");
            var rays = CreateRays();
            mTraversal.SetRays(rays);

        }
    }
}