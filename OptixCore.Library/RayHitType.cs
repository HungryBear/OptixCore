namespace OptixCore.Library
{
    /// <summary>
    /// Enum containing the different types of surface-ray intersections
    /// </summary>
    public enum RayHitType
    {
        /// <summary>
        /// Optix will return the first surface it intersects with Any hit ray types
        /// </summary>
        Any,

        /// <summary>
        /// Optix will return the closest surface it intersects with Closest hit ray types
        /// </summary>
        Closest
    };
}