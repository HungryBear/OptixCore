namespace OptixCore.Library
{
    public struct OptixDevice
    {
        public int Index;

        /// <summary>
        /// Maximum threads per block.
        /// </summary>
        public int MaxThreadsPerBlock;

        /// <summary>
        /// GPU clock rate.
        /// </summary>
        public int ClockRate;

        /// <summary>
        /// Streaming muli-processor count of the device.
        /// </summary>
        public int ProcessorCount;

        /// <summary>
        /// Is execution timeout enabled.
        /// </summary>
        public int ExecutionTimeOutEnabled;

        /// <summary>
        /// Maximum number of textures that can be created.
        /// </summary>
        public int MaxHardwareTextureCount;

        /// <summary>
        /// Name of the device.
        /// </summary>
        public string Name;

        /// <summary>
        /// Compute capability of the device.
        /// </summary>
        public Int2 ComputeCapability;

        /// <summary>
        /// Total memory of the device.
        /// </summary>
        public int TotalMemory;
    }
}