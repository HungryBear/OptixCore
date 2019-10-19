using System;
using System.Numerics;
using OptixCore.Library.Native.Prime;
using OptixCore.Library.Prime;

namespace OptixCore.PrimeSample
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var ctx = new PrimeContext())
            {
                //ctx.InitCuda();
                Console.WriteLine("Context is created");
                Simple(ctx);
                //Vector3[] vertices =
                //{
                //    new Vector3(0.0f),
                //    new Vector3(1.0f, 0.0f, 0.0f),
                //    new Vector3(1.0f, 1.0f, 0.0f),
                //};
                //using (var vertexBuffer = ctx.CreateBuffer(RTPBufferType.CudaLinear, RtpBufferFormat.VERTEX_FLOAT3, vertices))
                //{
                //    //vertexBuffer.SetRange(0, 3);

                //    //vertexBuffer.Lock();
                //    var data = vertexBuffer.GetData<Vector3>();
                //    foreach (var vector3 in data)
                //    {
                //        Console.WriteLine(vector3);
                //    }
                //    //vertexBuffer.Unlock();
                //}

                //Simple(ctx);
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

            using (var vertexBuffer = ctx.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.VERTEX_FLOAT3, vertices))
            {
                //vertexBuffer.SetRange(0, 3);
                var data = vertexBuffer.GetData<Vector3>();

                using (var indexBuffer = ctx.CreateBuffer(RTPBufferType.Host, RtpBufferFormat.IndicesInt3, indices))
                {
                    //indexBuffer.SetRange(0, 1);
                    using (var model = new PrimeModel(ctx))
                    {
                        model.SetTriangles(indexBuffer, vertexBuffer);
                        model.Update(0);

                        var r = new Ray
                        { origin = new Vector3(0.3f, 0.3f, -0.1f), dir = new Vector3(0, 0, 0.99f), tmax = 1e34f };

                        var rayHit = new Hit();
                        var hitData = new[] { rayHit };
                        using (var rayBuffer = ctx.CreateBuffer(RTPBufferType.CudaLinear,
                            RtpBufferFormat.RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, new[] { r }))
                        {
                            //rayBuffer.SetRange(0,1);

                            using (var hitBuffer =
                                ctx.CreateBuffer(RTPBufferType.CudaLinear,
                                    RtpBufferFormat.RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, hitData))
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
        }
    }
}
