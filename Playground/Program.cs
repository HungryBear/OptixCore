using System;
using OptixCore.Library;

namespace Playground
{
    class Program
    {
        static void Main(string[] args)
        {
            //using (var ctx = new Context())
            //{
            //    Console.WriteLine($"Hello World! Optix version is {ctx.GetOptixVersion()}");
            //    Console.WriteLine($"Devices count {ctx.GetDevicesCount()}");
            //    Console.WriteLine($"Available memory {ctx.AvailableMemory}");
            //    Console.WriteLine($"Max textures {ctx.MaxTextureCount}");
            //    Console.WriteLine($"Cpu threads {ctx.CpuThreadsCount}");
            //    Console.WriteLine($"Paging enabled {ctx.IsGpuPagingEnabled}");
            //    Console.WriteLine($"Paging force-disabled {ctx.IsGpuPagingForcedlyDisabled}");
            //}

            using (var window = new PathTracerWindow())
            {
                window.Run();
            }
        }
    }
}
