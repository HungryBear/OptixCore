namespace OptixCore.Library
{
    public interface IVariableContainer
    {
        Variable this[int index] { get; set; }
        Variable this[string name] { get; set; }

        int VariableCount { get; }
    }
}