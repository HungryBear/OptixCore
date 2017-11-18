using System;

namespace OptixCore.Library
{
    /// <summary>
    /// A <see cref="OptixProgram">Program</see> representing a Surface shader program. Called when a ray intersects a surface.
    /// </summary>
    public class SurfaceProgram : OptixProgram
    {

        /// <summary>
        /// Creates a program located at [filename] with main function [programName]
        /// </summary>
        /// <param name="context">Context with which to create the program.</param>
        /// <param name="type">Type of ray this program represents. <seealso cref="RayHitType"></seealso></param>
        /// <param name="filename">Path of the compiled cuda ptx file holding the program.</param>
        /// <param name="programName">Name of the main function of the program.</param>
        public SurfaceProgram(Context context, RayHitType type, String filename, String programName) :base(context, filename, programName)
        {
            RayType = type;
        }

        internal SurfaceProgram(Context ctx, RayHitType type, IntPtr program):base(ctx, program)
        {
            RayType = type;
        }

        public RayHitType RayType { get; }
    }
}