using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeModel : OptixPrimeNode
    {
        public PrimeModel(PrimeContext context) : base(context)
        {
            CheckError(PrimeApi.rtpModelCreate(context.InternalPtr, out InternalPtr));
        }

        public override void Validate()
        {
            
        }

        public void Update(uint hints)
        {
            CheckError(PrimeApi.rtpModelUpdate(InternalPtr, hints));
        }

        public override void Destroy()
        {
            CheckError(PrimeApi.rtpModelDestroy(InternalPtr));
        }
    }
}