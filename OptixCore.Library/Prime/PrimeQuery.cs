using System;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class PrimeQuery : OptixPrimeNode
    {
        private readonly QueryType _queryType;

        public PrimeQuery(PrimeContext context, PrimeModel model, QueryType queryType) : base(context)
        {
            _queryType = queryType;
            CheckError(PrimeApi.rtpQueryCreate(model.InternalPtr, (RTPquerytype)queryType, out InternalPtr));
        }

        public override void Validate()
        {
            
        }

        public void SetRays(PrimeBuffer rays)
        {
            CheckError(PrimeApi.rtpQuerySetRays(InternalPtr, rays.InternalPtr));
        }

        public void SetHits(PrimeBuffer hits)
        {
            CheckError(PrimeApi.rtpQuerySetHits(InternalPtr, hits.InternalPtr));
        }

        public void Execute(uint hints)
        {
            CheckError(PrimeApi.rtpQueryExecute(InternalPtr, hints));
        }

        public void SetCudaStream(IntPtr cudaStream)
        {
            CheckError(PrimeApi.rtpQuerySetCudaStream(InternalPtr, cudaStream ));
        }

        public void Finish()
        {
            CheckError(PrimeApi.rtpQueryFinish(InternalPtr));
        }

        public override void Destroy()
        {
            CheckError(PrimeApi.rtpQueryDestroy(InternalPtr));
        }
    }
}