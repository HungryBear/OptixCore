using System;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using OptixCore.Library;
using OptixCore.Library.Native;
using OptixCore.Library.Scene;

namespace Playground
{
    public class PathTracerWindow : OptixWindow
    {
        private uint mFrame = 0;

        //private float yAvg = 0.0125f;
        //private float yMax = 2.5f;

        public PathTracerWindow() : base(800, 600)
        {
            UseSRGB = true;
            UsePBO = true;

        }

        protected override void Initialize()
        {
            base.Initialize();
            var rayGenPath = GetScript("path_tracer.cu.ptx");
            var shaderPath = GetScript("path_tracer.cu.ptx");
            var modelName = "cornell-dragon.obj";
            var modelPath = Path.GetFullPath(@"..\..\..\..\..\Assets\Models\" + modelName);
            /*-----------------------------------------------
            * Create the Optix context
            *-----------------------------------------------*/
            OptixContext = new Context();
            OptixContext.RayTypeCount = 2;
            OptixContext.EntryPointCount = 1;
            OptixContext.EnableAllExceptions = false;

            /*-----------------------------------------------
             * Create the material that will be executed when there is an intersection
             *-----------------------------------------------*/
            Material CreateMaterial(string h, string a)
            {
                var newMat = new Material(OptixContext);
                Console.WriteLine("Shader path = " + shaderPath);
                newMat.Programs[0] = new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, h);
                newMat.Programs[1] = new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, a);
                return newMat;
            }

            var material = CreateMaterial("diffuse", "shadow");
            var glass = CreateMaterial("specular", "shadow");

            /*-----------------------------------------------
             * Load the geometry
             *-----------------------------------------------*/
            var model = new OptixOBJLoader(modelPath, OptixContext, null, material, n => n.Equals("01___Default") ? glass : null);
            model.GeoGroup = new GeometryGroup(OptixContext);
            model.ParseNormals = false;
            model.GenerateNormals = false;

            string intersectPath = GetScript("triangle_mesh.cu.ptx");
            model.IntersecitonProgPath = intersectPath;
            model.BoundingBoxProgPath = intersectPath;
            model.IntersecitonProgName = "mesh_intersect";
            model.BoundingBoxProgName = "mesh_bounds";

            model.LoadContent();

            /*-----------------------------------------------
             * Create scene lights
             *-----------------------------------------------*/
            CreateLights();

            /*-----------------------------------------------
             * Create the output buffer
             *-----------------------------------------------*/
            CreateOutputBuffer(Format.Float4);

            /*-----------------------------------------------
             * Create the ray-generation and exception programs
             *-----------------------------------------------*/
            var rayGen = new OptixProgram(OptixContext, rayGenPath, "pathtrace_camera");
            var exception = new OptixProgram(OptixContext, rayGenPath, "exception");
            var miss = new OptixProgram(OptixContext, shaderPath, "miss");
            miss["bg_color"].Set(100 / 255.0f, 149 / 255.0f, 237 / 255.0f);

            OptixContext.SetRayGenerationProgram(0, rayGen);
            OptixContext.SetExceptionProgram(0, exception);
            OptixContext.SetRayMissProgram(0, miss);

            /*-----------------------------------------------
             * Finally compile the optix context, and build the accel tree
             *-----------------------------------------------*/
            SetCamera(model.BBox);


            OptixContext["top_object"].Set(model.GeoGroup);
            OptixContext["output_buffer"].Set(OutputBuffer);
            OptixContext["scene_epsilon"].Set(0.0003f);

            const uint sqrtSamples = 2u;
            const uint maxDepth = 3u;
            const uint rrBeginDepth = 1u;

            OptixContext["rr_begin_depth"].Set(rrBeginDepth);
            OptixContext["max_depth"].Set(maxDepth);
            OptixContext["sqrt_num_samples"].Set(sqrtSamples);
            OptixContext["frame_number"].Set(mFrame);

            OptixContext["pathtrace_ray_type"].Set(0u);
            OptixContext["pathtrace_shadow_ray_type"].Set(1u);

            OptixContext["bad_color"].Set(1.0f, 0.0f, 0.0f);

            Trace.Write("Compiling Optix... ");

            OptixContext.Compile();
            OptixContext.BuildAccelTree();

            //give aproximate here becuase we'll have somewhere between rrBeginDepth and maxDepth number of iterations per sample.
            int avgIteration = (int)(rrBeginDepth + maxDepth) / 2;
            RaysTracedPerFrame = Width * Height * 2 * ((int)sqrtSamples * (int)sqrtSamples) * avgIteration;

        }

        private void CreateLights()
        {
            ParallelogramLight light = new ParallelogramLight();
            light.corner = new Vector3(2.8096f, 17.1165f, 2.3659f);
            light.v1 = new Vector3(-5.0f, 0.0f, 0.0f);
            light.v2 = new Vector3(0.0f, 0.0f, 5.0f);

            light.normal = Vector3.Normalize(Vector3.Cross(light.v1, light.v2));
            light.emission = new Vector3(15.0f, 15.0f, 5.0f);

            var desc = new BufferDesc() { Width = 1u, Format = Format.User, Type = BufferType.Input, ElemSize = (uint)Marshal.SizeOf(typeof(ParallelogramLight)) };
            var lightBuffer = new OptixBuffer(OptixContext, desc);
            BufferStream stream = lightBuffer.Map();
            stream.Write(light);
            lightBuffer.Unmap();

            OptixContext["lights"].Set(lightBuffer);
        }

        private void SetCamera(BoundingBox box)
        {
            Camera = new Camera();
            Camera.Aspect = (float)Width / (float)Height;
            Camera.Fov = 35.0f;
            Camera.RotationVel = 100.0f;
            Camera.TranslationVel = 100.0f;

            Camera.CenterOnBoundingBox(box, .95f);

            CameraUpdate();
        }

        protected override void RayTrace()
        {
            base.RayTrace();

            OptixContext["frame_number"].Set(mFrame++);
            OptixContext.Launch(0, (uint)Width, (uint)Height);
        }

        protected override void CameraUpdate()
        {
            mFrame = 0;
            Console.WriteLine($"Camera = {Camera.Position}f  {Camera.Target}f  {Camera.Up}f");
            var eye = Camera.Position;
            var right = Camera.Right;
            var up = Camera.Up;
            var look = Camera.Look;

            OptixContext["eye"].Set(ref eye);
            OptixContext["U"].Set(ref right);
            OptixContext["V"].Set(ref up);
            OptixContext["W"].Set(ref look);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct ParallelogramLight
        {
            public Vector3 corner;
            public Vector3 v1, v2;
            public Vector3 normal;
            public Vector3 emission;
            public int textured;
        };
    }
}