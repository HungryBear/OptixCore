namespace OptixCore.Library
{
    public interface INodeCollectionProvider<T>
    {
        NodeCollection<T> Collection { get; }
    }
}