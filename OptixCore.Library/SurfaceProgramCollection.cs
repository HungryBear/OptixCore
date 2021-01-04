namespace OptixCore.Library
{
    /// <summary>
    /// A collection of <see cref="SurfaceProgram">SurfacePrograms</see>. Allows one to iterate through the SurfacePrograms contained by a DefaultMaterial.
    /// </summary>
    public class SurfaceProgramCollection
    {
        private readonly Material _container;

        internal SurfaceProgramCollection(Material container)
        {
            _container = container;
        }

        /// <summary>
        /// Gets the number of ray types set on the parent Context
        /// </summary>
        public uint Count => _container.mContext.RayTypeCount;

        public SurfaceProgram this[int index]
        {
            get => _container.GetSurfaceProgram(index);
            set => _container.SetSurfaceProgram(index, value);
        }
    }
}