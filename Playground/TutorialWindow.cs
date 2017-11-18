using System;
using System.DrawingCore;
using System.DrawingCore.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using OpenTK;
using OptixCore.Library;
using OptixCore.Library.Native;
using OptixCore.Library.Scene;
using Bitmap = System.DrawingCore.Bitmap;
using Image = System.DrawingCore.Image;
using Vector3 = System.Numerics.Vector3;
using Vector4 = System.Numerics.Vector4;

namespace Playground
{
    using Buffer = OptixBuffer;

    public class TutorialWindow : OptixWindow
    {
        public struct BasicLight
        {
            public Vector3 Position;
            public Vector3 Color;
            public int CastsShadow;
#pragma warning disable 169
            int padding;      // make this structure 32 bytes -- powers of two are your friend!
#pragma warning restore 169
        }

        private static int mTutorial = 10;
        private string EnvMapPath = @"G:\Depot\Source\OptixDotNet\Assets\Textures\CedarCity.png";
        private string shaderPath = GetScript($"tutorial{mTutorial}.cu.ptx");
        string boxPath = GetScript("box.cu.ptx");
        string parrallelPath = GetScript("parallelogram.cu.ptx");

        public TutorialWindow() : base(800, 600)
        {
        }

        protected override void Initialize()
        {
            string rayGenPath = shaderPath;

            /*-----------------------------------------------
             * Create the Optix context
             *-----------------------------------------------*/
            OptixContext = new Context();
            OptixContext.RayTypeCount = 2;
            OptixContext.EntryPointCount = 1;
            OptixContext.SetStackSize(4096);

            /* Create the ray-generation and exception programs
             *-----------------------------------------------*/
            var rayGen = new OptixProgram(OptixContext, rayGenPath, mTutorial < 11 ? "pinhole_camera" : "env_camera");
            var exception = new OptixProgram(OptixContext, rayGenPath, "exception");
            var miss = new OptixProgram(OptixContext, rayGenPath, mTutorial < 5 ? "miss" : "envmap_miss");
            //miss["bg_color"].Set(100 / 255.0f, 149 / 255.0f, 237 / 255.0f);
            //exception["bad_color"].Set(1.0f, 0.0f, 0.0f);

            OptixContext.SetRayGenerationProgram(0, rayGen);
            OptixContext.SetExceptionProgram(0, exception);
            OptixContext.SetRayMissProgram(0, miss);

            /*-----------------------------------------------
             * Create lights
             *-----------------------------------------------*/
            BasicLight[] lights = new BasicLight[1];
            lights[0].Position = new Vector3(-5.0f, 60.0f, -16.0f);
            lights[0].Color = new Vector3(1.0f, 1.0f, 1.0f);
            lights[0].CastsShadow = 1;

            BufferDesc desc = new BufferDesc()
            {
                Width = (uint)lights.Length,
                Format = Format.User,
                Type = BufferType.Input,
                ElemSize = (uint)Marshal.SizeOf(typeof(BasicLight))
            };
            Buffer lightsBuffer = new Buffer(OptixContext, desc);
            lightsBuffer.SetData<BasicLight>(lights);

            OptixContext["lights"].Set(lightsBuffer);

            /*-----------------------------------------------
          * Create noise texture
          *-----------------------------------------------*/
            if (mTutorial >= 8)
            {
                uint noiseTexDim = 64;
                desc = new BufferDesc() { Width = noiseTexDim, Height = noiseTexDim, Depth = noiseTexDim, Format = Format.Float, Type = BufferType.Input };
                Buffer noiseBuffer = new Buffer(OptixContext, desc);

                Random rand = new Random();
                BufferStream stream = noiseBuffer.Map();
                for (int i = 0; i < noiseTexDim * noiseTexDim * noiseTexDim; i++)
                {
                    stream.Write<float>((float)rand.NextDouble());
                }
                noiseBuffer.Unmap();

                TextureSampler noiseTex = new TextureSampler(OptixContext, TextureSamplerDesc.GetDefault(WrapMode.Repeat));
                noiseTex.SetBuffer(noiseBuffer);

                OptixContext["noise_texture"].Set(noiseTex);
            }

            /*-----------------------------------------------
             * Load enivronment map texture
             *-----------------------------------------------*/
            LoadEnvMap();

            /*-----------------------------------------------
             * Load the geometry
             *-----------------------------------------------*/
            CreateGeometry();

            /*-----------------------------------------------
             * Create the output buffer
             *-----------------------------------------------*/
            CreateOutputBuffer(Format.UByte4);
            OptixContext["output_buffer"].Set(OutputBuffer);

            /*-----------------------------------------------
            * Finally compile the optix context, and build the accel tree
            *-----------------------------------------------*/
            SetCamera();

            OptixContext["max_depth"].Set(100);
            OptixContext["radiance_ray_type"].Set(0u);
            OptixContext["shadow_ray_type"].Set(1u);
            OptixContext["frame_number"].Set(0u);
            OptixContext["scene_epsilon"].Set(.001f);
            OptixContext["importance_cutoff"].Set(0.01f);
            OptixContext["ambient_light_color"].Set(0.31f, 0.33f, 0.28f);

            OptixContext.Compile();
            OptixContext.BuildAccelTree();

            //very loose calculation of number of rays
            float numSecondaryRays = 0;
            if (mTutorial >= 9)
                numSecondaryRays = 2.5f; //only convex hull casts refraction rays
            else if (mTutorial >= 8)
                numSecondaryRays = 2;
            else if (mTutorial >= 4)
                numSecondaryRays = 1.5f; //only the floor casts reflection rays, so give aproximate
            else if (mTutorial >= 3)
                numSecondaryRays = 1;

            RaysTracedPerFrame = (int)(Width * Height * (numSecondaryRays + 1));
        }

        protected override void RayTrace()
        {
            base.RayTrace();

            OptixContext.Launch(0, (uint)Width, (uint)Height);
        }
        private void SetCamera()
        {
            Camera = new Camera();
            Camera.Aspect = (float)Width / (float)Height;
            Camera.Fov = 75.0f;
            Camera.RotationVel = 100.0f;
            Camera.TranslationVel = 500.0f;

            Camera.LookAt(new Vector3(7.0f, 9.2f, -6.0f), new Vector3(0.0f, 4.0f, 0.0f), Vector3.UnitY);

            CameraUpdate();
        }

        private void CreateGeometry()
        {
            // Create box
            Geometry box = new Geometry(OptixContext);
            box.PrimitiveCount = 1;
            box.BoundingBoxProgram = new OptixProgram(OptixContext, boxPath, "box_bounds"); ;
            box.IntersectionProgram = new OptixProgram(OptixContext, boxPath, "box_intersect"); ;
            box["boxmin"].Set(-2.0f, 0.0f, -2.0f);
            box["boxmax"].Set(2.0f, 7.0f, 2.0f);

            Geometry chull = null;
            if (mTutorial >= 9)
            {
                chull = new Geometry(OptixContext);
                chull.PrimitiveCount = 1u;
                chull.BoundingBoxProgram = new OptixProgram(OptixContext, shaderPath, "chull_bounds");
                chull.IntersectionProgram = new OptixProgram(OptixContext, shaderPath, "chull_intersect");

                uint nsides = 6;
                float radius = 1;
                Vector3 xlate = new Vector3(-1.4f, 0, -3.7f);

                BufferDesc desc = new BufferDesc() { Width = nsides + 2u, Type = BufferType.Input, Format = Format.Float4 };
                Buffer chullBuffer = new Buffer(OptixContext, desc);

                float angle = 0.0f;
                BufferStream stream = chullBuffer.Map();
                for (uint i = 0; i < nsides; i++)
                {
                    angle = (float)i / (float)nsides * (float)Math.PI * 2.0f;
                    float x = (float)Math.Cos(angle);
                    float y = (float)Math.Sin(angle);

                    stream.Write<Vector4>(Utils.CreatePlane(new Vector3(x, 0, y), new Vector3(x * radius, 0, y * radius) + xlate));
                }

                float min = 0.02f;
                float max = 3.5f;
                angle = 5.0f / (float)nsides * (float)Math.PI * 2.0f;
                stream.Write<Vector4>(Utils.CreatePlane(new Vector3(0, -1, 0), new Vector3(0, min, 0) + xlate));
                stream.Write<Vector4>(Utils.CreatePlane(new Vector3((float)Math.Cos(angle), 0.7f, (float)Math.Sin(angle)),
                                                            new Vector3(0, max, 0) + xlate));

                chullBuffer.Unmap();

                chull["planes"].Set(chullBuffer);
                chull["chull_bbmin"].Set(-radius + xlate.X, min + xlate.Y, -radius + xlate.Z);
                chull["chull_bbmax"].Set(radius + xlate.X, max + xlate.Y, radius + xlate.Z);
            }

            // Floor geometry
            Geometry parallelogram = new Geometry(OptixContext);
            parallelogram.PrimitiveCount = 1;
            parallelogram.BoundingBoxProgram = new OptixProgram(OptixContext, parrallelPath, "bounds"); ;
            parallelogram.IntersectionProgram = new OptixProgram(OptixContext, parrallelPath, "intersect"); ;

            Vector3 anchor = new Vector3(-64.0f, 0.01f, -64.0f);
            Vector3 v1 = new Vector3(128.0f, 0.0f, 0.0f);
            Vector3 v2 = new Vector3(0.0f, 0.0f, 128.0f);
            Vector3 normal = Vector3.Normalize(Vector3.Cross(v2, v1));

            v1 *= 1.0f / (v1.LengthSquared());
            v2 *= 1.0f / (v2.LengthSquared());

            float d = Vector3.Dot(normal, anchor);
            Vector4 plane = new Vector4(normal, d);

            parallelogram["plane"].Set(ref plane);
            parallelogram["v1"].Set(ref v1);
            parallelogram["v2"].Set(ref v2);
            parallelogram["anchor"].Set(ref anchor);

            string boxMtrlName = mTutorial >= 8 ? "box_closest_hit_radiance" : "closest_hit_radiance";
            string floorMtrlName = mTutorial >= 4 ? "floor_closest_hit_radiance" : "closest_hit_radiance";

            Material boxMtrl = new Material(OptixContext);
            boxMtrl.SetSurfaceProgram(0, new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, boxMtrlName));
            if (mTutorial >= 3)
                boxMtrl.SetSurfaceProgram(1, new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, "any_hit_shadow"));

            boxMtrl["Ka"].Set(0.3f, 0.3f, 0.3f);
            boxMtrl["Kd"].Set(0.6f, 0.7f, 0.8f);
            boxMtrl["Ks"].Set(0.8f, 0.9f, 0.8f);
            boxMtrl["phong_exp"].Set(88.0f);
            boxMtrl["reflectivity_n"].Set(0.2f, 0.2f, 0.2f);

            Material floorMtrl = new Material(OptixContext);
            floorMtrl.SetSurfaceProgram(0, new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, floorMtrlName));
            if (mTutorial >= 3)
                floorMtrl.SetSurfaceProgram(1, new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, "any_hit_shadow"));

            floorMtrl["Ka"].Set(0.3f, 0.3f, 0.1f);
            floorMtrl["Kd"].Set(194 / 255.0f * .6f, 186 / 255.0f * .6f, 151 / 255.0f * .6f);
            floorMtrl["Ks"].Set(0.4f, 0.4f, 0.4f);
            floorMtrl["reflectivity"].Set(0.1f, 0.1f, 0.1f);
            floorMtrl["reflectivity_n"].Set(0.05f, 0.05f, 0.05f);
            floorMtrl["phong_exp"].Set(88.0f);
            floorMtrl["tile_v0"].Set(0.25f, 0, .15f);
            floorMtrl["tile_v1"].Set(-.15f, 0, 0.25f);
            floorMtrl["crack_color"].Set(0.1f, 0.1f, 0.1f);
            floorMtrl["crack_width"].Set(0.02f);

            Material glassMtrl = null;
            if (chull != null)
            {
                glassMtrl = new Material(OptixContext);
                glassMtrl.SetSurfaceProgram(0, new SurfaceProgram(OptixContext, RayHitType.Closest, shaderPath, "glass_closest_hit_radiance"));
                glassMtrl.SetSurfaceProgram(1, new SurfaceProgram(OptixContext, RayHitType.Any, shaderPath, mTutorial > 9 ? "glass_any_hit_shadow" : "any_hit_shadow"));

                Vector3 extinction = new Vector3(.80f, .89f, .75f);
                glassMtrl["importance_cutoff"].Set(1e-2f);
                glassMtrl["cutoff_color"].Set(0.34f, 0.55f, 0.85f);
                glassMtrl["fresnel_exponent"].Set(3.0f);
                glassMtrl["fresnel_minimum"].Set(0.1f);
                glassMtrl["fresnel_maximum"].Set(1.0f);
                glassMtrl["refraction_index"].Set(1.4f);
                glassMtrl["refraction_color"].Set(1.0f, 1.0f, 1.0f);
                glassMtrl["reflection_color"].Set(1.0f, 1.0f, 1.0f);
                glassMtrl["refraction_maxdepth"].Set(100);
                glassMtrl["reflection_maxdepth"].Set(100);
                glassMtrl["extinction_constant"].Set((float)Math.Log(extinction.X), (float)Math.Log(extinction.Y), (float)Math.Log(extinction.Z));
                glassMtrl["shadow_attenuation"].Set(0.4f, 0.7f, 0.4f);
            }

            GeometryInstance boxInst = new GeometryInstance(OptixContext);
            boxInst.Geometry = box;
            boxInst.AddMaterial(boxMtrl);

            GeometryInstance parallelInst = new GeometryInstance(OptixContext);
            parallelInst.Geometry = parallelogram;
            parallelInst.AddMaterial(floorMtrl);

            GeometryGroup group = new GeometryGroup(OptixContext);
            group.AddChild(boxInst);
            group.AddChild(parallelInst);
            if (chull != null)
            {
                GeometryInstance chullInst = new GeometryInstance(OptixContext);
                chullInst.Geometry = chull;
                chullInst.AddMaterial(glassMtrl);
                group.AddChild(chullInst);
            }

            group.Acceleration = new Acceleration(OptixContext, AccelBuilder.Bvh, AccelTraverser.Bvh);

            OptixContext["top_object"].Set(group);
            OptixContext["top_shadower"].Set(group);
        }
        private void LoadEnvMap()
        {
            if (mTutorial < 5)
                return;

            string texturePath = Path.GetFullPath(EnvMapPath);

            Bitmap image = new Bitmap(Image.FromFile(texturePath));

            if (image == null)
                return;

            var imgData = image.LockBits(new System.DrawingCore.Rectangle(0, 0, image.Width, image.Height),
                                                ImageLockMode.ReadWrite, image.PixelFormat);


            BufferDesc bufferDesc = new BufferDesc() { Width = (uint)image.Width, Height = (uint)image.Height, Format = Format.UByte4, Type = BufferType.Input };
            Buffer textureBuffer = new Buffer(OptixContext, bufferDesc);

            int stride = imgData.Stride;
            int numChannels = 4;

            unsafe
            {
                byte* src = (byte*)imgData.Scan0.ToPointer();

                BufferStream stream = textureBuffer.Map();
                for (int h = 0; h < image.Height; h++)
                {
                    for (int w = 0; w < image.Width; w++)
                    {
                        UByte4 color = new UByte4(src[(image.Height - h - 1) * stride + w * numChannels + 2],
                                                   src[(image.Height - h - 1) * stride + w * numChannels + 1],
                                                   src[(image.Height - h - 1) * stride + w * numChannels + 0],
                                                   255);

                        stream.Write<UByte4>(color);
                    }
                }
                textureBuffer.Unmap();
            }
            image.UnlockBits(imgData);

            TextureSampler texture = new TextureSampler(OptixContext, TextureSamplerDesc.GetDefault(WrapMode.Repeat));
            texture.SetBuffer(textureBuffer);

            OptixContext["envmap"].Set(texture);
        }
        protected override void CameraUpdate()
        {
            Vector3 eye = Camera.Position;
            Vector3 right = Camera.Right;
            Vector3 up = Camera.Up;
            Vector3 look = Camera.Look;

            OptixContext["eye"].Set(ref eye);
            OptixContext["U"].Set(ref right);
            OptixContext["V"].Set(ref up);
            OptixContext["W"].Set(ref look);
        }
        private OptixBuffer CreateBuffer(float[] data)
        {
            BufferDesc spdBuffDesc = new BufferDesc
            {
                Width = (ulong)data.Length,
                Format = Format.Float,
                //ElemSize = 0,
                Type = BufferType.Input
            };
            var x = new OptixBuffer(OptixContext, spdBuffDesc);
            x.SetData(data);
            return x;
        }
    }
}