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
    public class VolumePathTracerWindow : OptixWindow
    {
        private uint mFrame = 0;
        private const uint sqrtSamples = 2u;
        private const uint maxDepth = 8u;
        private const uint rrBeginDepth = 3u;
        string rayGenPath;
        string shaderPath;

        public VolumePathTracerWindow() : base(800, 600)
        {
            UseSRGB = true;
            UsePBO = false;

        }

        protected override void Initialize()
        {
            base.Initialize();
            rayGenPath = GetScript("vpt.cu.ptx");
            shaderPath = GetScript("vpt.cu.ptx");
            var modelName = "cornell-dragon.obj";
            var modelPath = Path.GetFullPath(@"..\Assets\Models\" + modelName);
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
            var material = new Material(OptixContext);

            material.Programs[0] = new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, "diffuse");
            material.Programs[1] = new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, "shadow");

            /*-----------------------------------------------
             * Load the geometry
             *-----------------------------------------------*/
            var model = new OptixOBJLoader(modelPath, OptixContext, null, material);
            model.GeoGroup = new GeometryGroup(OptixContext);
            model.ParseNormals = false;
            model.GenerateNormals = false;

            string intersectPath = GetScript("triangle_mesh.cu.ptx");
            model.IntersecitonProgPath = intersectPath;
            model.BoundingBoxProgPath = intersectPath;
            model.IntersecitonProgName = "mesh_intersect";
            model.BoundingBoxProgName = "mesh_bounds";

            model.LoadContent(this.resolveMaterial);

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
            var avgIteration = (int)(rrBeginDepth + maxDepth) / 2;
            RaysTracedPerFrame = Width * Height * 2 * ((int)sqrtSamples * (int)sqrtSamples) * avgIteration;

        }

        private Material resolveMaterial(string objName, string matName)
        {
            var material = new Material(OptixContext);

            material.Programs[0] = new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, matName.Contains("01___Default") ? "specular" : "diffuse");
            material.Programs[1] = new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, "shadow");

            return material;
        }

        private void CreateLights()
        {
            var light = new ParallelogramLight
            {
                corner = new Vector3(2.8096f, 17.1165f, 2.3659f),
                v1 = new Vector3(-5.0f, 0.0f, 0.0f),
                v2 = new Vector3(0.0f, 0.0f, 5.0f)
            };

            light.normal = Vector3.Normalize(Vector3.Cross(light.v1, light.v2));
            light.emission = new Vector3(15.0f, 15.0f, 5.0f);

            var desc = new BufferDesc { Width = 1u, Format = Format.User, Type = BufferType.Input, ElemSize = (uint)Marshal.SizeOf(typeof(ParallelogramLight)) };
            var lightBuffer = new OptixBuffer(OptixContext, desc);
            var stream = lightBuffer.Map();
            stream.Write(light);
            lightBuffer.Unmap();

            OptixContext["lights"].Set(lightBuffer);
        }

        private void SetCamera(BoundingBox box)
        {
            Camera = new Camera();
            Camera.Aspect = (float)Width / (float)Height;
            Camera.Fov = 60.0f * MathF.PI / 180.0f;
            Camera.RotationVel = 100.0f;
            Camera.TranslationVel = 100.0f;

            Camera.CenterOnBoundingBox(box, 1.95f);
            //Camera.Position = new Vector3(0, 8.5f, 260.0f);
            var optixCamera = new OptixCamera();
            var cameraPosition = Camera.Position;
            OptixContext["camPos"].Set(ref cameraPosition);
            var cameraTarget = Camera.Target;
            OptixContext["camForward"].Set(ref cameraTarget);
            var cameraUp = Camera.Up;
            OptixContext["camUp"].Set(ref cameraUp);
            OptixContext["fov"].Set(Camera.Fov);
            optixCamera.Setup(Camera.Position, Camera.Look, Camera.Up, new Vector2(Width, Height), Camera.Fov);
            OptixContext["mPosition"].SetFloat3(optixCamera.mPosition);
            OptixContext["mForward"].SetFloat3(optixCamera.mForward);
            OptixContext["mResolution"].Set(optixCamera.mResolution.X, optixCamera.mResolution.Y);
            OptixContext["mRasterToWorld"].Set(ref optixCamera.mRasterToWorld);
            OptixContext["mWorldToRaster"].Set(ref optixCamera.mWorldToRaster);
            OptixContext["mImagePlaneDist"].Set(optixCamera.mImagePlaneDist);

            CameraUpdate();
        }


        public class OptixCamera
        {
            internal Vector3 mPosition;
            internal Vector3 mForward;
            internal Vector2 mResolution;
            internal Matrix4x4 mRasterToWorld;
            internal Matrix4x4 mWorldToRaster;
            internal float mImagePlaneDist;

            public void Setup( 
            Vector3 aPosition,
            Vector3 aForward,
            Vector3 aUp,
            Vector2 aResolution,
            float aHorizontalFOV)
            {

                var forward = Vector3.Normalize(aForward);
                var up = Vector3.Normalize(Vector3.Cross(aUp, -forward));
                var left = -Vector3.Cross(-forward, up);

                mPosition = aPosition;
                mForward = forward;
                mResolution = aResolution;

                var pos = new Vector3(
                    Vector3.Dot(up, aPosition),
                    Vector3.Dot(left, aPosition),
                    Vector3.Dot(-forward, aPosition));

                var worldToCamera = Matrix4x4.Identity;
                SetRow(ref worldToCamera, 0, new Vector4(up, -pos.X));
                SetRow(ref worldToCamera, 1, new Vector4(left, -pos.Y));
                SetRow(ref worldToCamera, 2, new Vector4(-forward, -pos.Z));

                var perspective = Perspective(aHorizontalFOV *MathF.PI / 180.0f,  0.1f, 10000f, aResolution.X / aResolution.Y);
                var worldToNScreen = perspective * worldToCamera;
                Matrix4x4.Invert(worldToNScreen, out var nscreenToWorld);
                

                mWorldToRaster =
                    Matrix4x4.CreateScale(new Vector3(aResolution.X * 0.5f, aResolution.Y * 0.5f, 0)) *
                    Matrix4x4.CreateTranslation(new Vector3(1f, 1f, 0)) * worldToNScreen;

                mRasterToWorld = nscreenToWorld *
                                 Matrix4x4.CreateTranslation(new Vector3(-1f, -1f, 0)) *
                                 Matrix4x4.CreateScale(new Vector3(2f / aResolution.X, 2f / aResolution.Y, 0));

                var tanHalfAngle = MathF.Tan(aHorizontalFOV *MathF.PI / 360f);
                mImagePlaneDist = aResolution.X / (2f * tanHalfAngle);
            }

            private Matrix4x4 Perspective(
                float aFov,
                float aNear,
                float aFar,
                float aspect)
            {
                // Camera points towards -z.  0 < near < far.
                // Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
                float f = 1f / (MathF.Tan(aFov * MathF.PI / 360.0f));
                float d = 1f / (aNear - aFar);

                Matrix4x4 r;
                r.M11 = f / aspect; r.M12 = 0.0f; r.M13 = 0.0f; r.M14 = 0.0f;
                r.M21 = 0.0f; r.M22 = -f; r.M23 = 0.0f; r.M24 = 0.0f;
                r.M31 = 0.0f; r.M32 = 0.0f; r.M33 = (aNear + aFar) * d; r.M34 = 2.0f * aNear * aFar * d;
                r.M41 = 0.0f; r.M42 = 0.0f; r.M43 = -1.0f; r.M44 = 0.0f;

                return r;
            }

            private void SetRow(ref Matrix4x4 m, int index, Vector4 row)
            {
                switch (index)
                {
                    case 0:
                        m.M11 = row.X;
                        m.M12 = row.Y;
                        m.M13 = row.Z;
                        m.M14 = row.W;
                        break;
                    case 1:
                        m.M21 = row.X;
                        m.M22 = row.Y;
                        m.M23 = row.Z;
                        m.M24 = row.W;
                        break;
                    case 2:
                        m.M31 = row.X;
                        m.M32 = row.Y;
                        m.M33 = row.Z;
                        m.M34 = row.W;
                        break;
                    case 3:
                        m.M41 = row.X;
                        m.M42 = row.Y;
                        m.M43 = row.Z;
                        m.M44 = row.W;
                        break;
                }
            }
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