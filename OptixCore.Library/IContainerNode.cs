namespace OptixCore.Library
{
    public interface IContainerNode<T>
    {
        T this[int index] {get;set;}
        int ChildCount { get; set; }
    }
}