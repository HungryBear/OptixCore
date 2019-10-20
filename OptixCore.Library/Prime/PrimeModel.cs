using System;
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

        public void SetTriangles(PrimeBuffer indices, PrimeBuffer vertices)
        {
            CheckError(PrimeApi.rtpModelSetTriangles(InternalPtr, 
                indices?.InternalPtr ?? IntPtr.Zero, 
                vertices.InternalPtr));
        }

        public void SetInstances(PrimeBuffer instances, PrimeBuffer transforms)
        {
            CheckError(PrimeApi.rtpModelSetInstances(InternalPtr, instances.InternalPtr, transforms.InternalPtr));
        }

        public void Finish()
        {
            CheckError(PrimeApi.rtpModelFinish(InternalPtr));
        }

        internal void Update(RTPmodelhint hints)
        {
            CheckError(PrimeApi.rtpModelUpdate(InternalPtr, (uint)hints));
        }

        public override void Destroy()
        {
            CheckError(PrimeApi.rtpModelDestroy(InternalPtr));
        }
    }
}