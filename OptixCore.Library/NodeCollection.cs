namespace OptixCore.Library
{
    public class NodeCollection<T>
    {
        protected IContainerNode<T> Container;
        public NodeCollection()
        {
            
        }
        public NodeCollection(IContainerNode<T> container )
        {
            Container = container;
        }

        public int Count
        {
            get => Container.ChildCount;
            set => Container.ChildCount = value;
        }

        public T this[int index]
        {
            get => Container[index];
            set => Container[index] = value;
        }
    }
}