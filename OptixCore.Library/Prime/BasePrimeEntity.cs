using System;
using OptixCore.Library.Native.Prime;

namespace OptixCore.Library.Prime
{
    public class BasePrimeEntity
    {
        protected internal IntPtr InternalPtr;

        internal void CheckError(RTPresult result)
        {
            if (result != RTPresult.RTP_SUCCESS)
            {
                try
                {
                    PrimeApi.rtpContextGetLastErrorString(InternalPtr, out var Errormessage);
                    //PrimeApi.rtpGetErrorString(result, out var message);
                    throw new OptixException($"Optix context error : {Errormessage}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Result {result} -Error getting error from Optix - " + ex);
                }
            }
        }
    }
}