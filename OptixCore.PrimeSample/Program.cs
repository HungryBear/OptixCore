using System;
using System.IO;
using System.Linq;
using System.Numerics;
using OptixCore.Library;
using OptixCore.Library.Native.Prime;
using OptixCore.Library.Prime;
using OptixCore.Library.Scene;
using RayFormat = OptixCore.Library.Prime.RayFormat;

namespace OptixCore.PrimeSample
{
    class Program
    {
        static void Main(string[] args)
        {
         /*   using (var context = new PrimeContext(true))
                Simple(context);*/

         IntersectionTest();
        }

        private static void IntersectionTest()
        {
            ICamera Camera;
            int Width = 320, Height = 200;

            Camera = new Camera();
            Camera.Aspect = (float) Width / (float) Height;
            Camera.Fov = 30;
            Camera.RotationVel = 100.0f;
            Camera.TranslationVel = 500.0f;

            //sibenik camera position
            //Camera.LookAt(new Vector3(-19.5f, -10.3f, .8f), new Vector3(0.0f, -13.3f, .8f), Vector3.UnitY);


            PrimeEngine.Ray[] CreateRays()
            {
                var rays = new PrimeEngine.Ray[Width * Height];

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

            using (var engine = new PrimeEngine(RayFormat.OriginDirectionMinMaxInterleaved, RayHitType.Closest, RTPBufferType.CudaLinear, false))
            {
                var modelName = "teapot.obj";
                var modelPath = Path.GetFullPath(@"..\..\..\..\Assets\Models\" + modelName);
                var model = new OBJLoader(modelPath);

                model.ParseMaterials = false;
                model.ParseNormals = false;
                model.GenerateNormals = false;
                model.GenerateGeometry = false;
                model.LoadContent();
                Camera.CenterOnBoundingBox(model.BBox);

                var verts = new Vector3[model.Positions.Count];
                var tris = new Int3[model.Groups[0].VIndices.Count];

                model.Positions.CopyTo(verts);
                model.Groups[0].VIndices.CopyTo(tris);
                var indexes = tris.SelectMany(c => new[] {Math.Abs(c.X), Math.Abs(c.Y), Math.Abs(c.Z)}).ToArray();
                Console.WriteLine("Setting Mesh");
                engine.SetMesh(verts, indexes);
                Console.WriteLine("Setting Rays");
                engine.SetRays(CreateRays());
                Console.WriteLine("Querying Prime");
                var hits = engine.Query();
                Console.WriteLine($"Successful hits {hits.Count(p => p.t < 1e34f && p.t > 1e-4f)}");
            }
        }


        static void Simple(PrimeContext ctx)
        {
            Vector3[] vertices = {
                    new Vector3(0.0f),
                    new Vector3(1.0f, 0.0f, 0.0f),
                    new Vector3(1.0f, 1.0f, 0.0f),
                };
            int[] indices = { 0, 2, 1 };

            var vertexBuffer = ctx.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.VERTEX_FLOAT3, vertices);
            vertexBuffer.SetRange(0, 3);
            var indexBuffer = ctx.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.IndicesInt3, indices);
            indexBuffer.SetRange(0, 1);
            using (var model = new PrimeModel(ctx))
            {
                model.SetTriangles(indexBuffer, vertexBuffer);
                model.Update(0);

                var r = new Ray
                { origin = new Vector3(0.3f, 0.3f, -0.1f), dir = new Vector3(0, 0, 0.99f), tmax = 1e34f };

                var rayHit = new Hit();
                var hitData = new[] { rayHit };
                var rayBuffer = ctx.CreateBuffer(RTPBufferType.CudaLinear,
                    RtpBufferFormat.RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, new[] {r});
                
                    var hitBuffer = ctx.CreateBuffer(RTPBufferType.CudaLinear, RtpBufferFormat.RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, hitData);
                    {
                        using (var query = new PrimeQuery(ctx, model, QueryType.AnyHit))
                        {
                            query.SetRays(rayBuffer);
                            query.SetHits(hitBuffer);
                            query.Execute(0);
                            query.Finish();


                            foreach (var hit in hitBuffer.GetData<Hit>())
                            {
                                if (hit.t > 0 && hit.t < 1e34f)
                                {
                                    Console.ForegroundColor = ConsoleColor.Green;
                                    Console.WriteLine("Hit!");
                                }
                                else
                                {
                                    Console.WriteLine("no hit");
                                }
                            }
                        }
                }
            }
        }
    }
}
