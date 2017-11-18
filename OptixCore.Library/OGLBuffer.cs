using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public struct OGLBufferDesc
    {
        /// <summary>
        /// Width of the OptixBuffer.
        /// </summary>
        public ulong Width;

        /// <summary>
        /// Height of the OptixBuffer.
        /// </summary>
        public ulong Height;

        /// <summary>
        /// Depth of the OptixBuffer.
        /// </summary>
        public ulong Depth;

        /// <summary>
        /// Size of the Type the buffer contains. Only needed for <see cref="Format.User">Format.User</see> formats.
        /// </summary>
        public ulong ElemSize;

        /// <summary>
        /// <see cref="BufferType">Type</see> of the OptixBuffer.
        /// </summary>
        public BufferType Type;

        /// <summary>
        /// <see cref="Format">Format</see> of the OptixBuffer
        /// </summary>
        public Format Format;

        /// <summary>
        /// ID of the OpenGl resource used to create the OGLBuffer.
        /// </summary>
        public uint Resource;


        public OGLBufferDesc(ulong width, ulong height, ulong depth, ulong elemsize, BufferType type, Format format, uint rc)
        {
            Width = width;
            Height = height;
            Depth = depth;
            ElemSize = elemsize;
            Type = type;
            Format = format;
            Resource = rc;
        }

        /// <summary>
        /// Gets a default BufferDesc with safe initilaized values.
        /// Width = 1
        /// Height = 1
        /// Depth = 1
        /// ElemSize = 0
        /// Type = None
        /// Format = Unknown
        /// </summary>
        public static OGLBufferDesc Default
        {
            get
            {
                var desc = new OGLBufferDesc(1, 1, 1, 0, BufferType.None, Format.Unknown, 0);
                return desc;
            }
        }
    };

    public class OGLBuffer : OptixBuffer
    {
        public uint ResourceId
        {
            get
            {
                CheckError(GlInterop.rtBufferGetGLBOId(InternalPtr,out var id));
                return id;
            }
        }
 
        public OGLBuffer(Context c, OGLBufferDesc desc) : base(c)
        {
            Create(desc);
        }

        private void Create(OGLBufferDesc desc)
        {

            mFormat = desc.Format;
            mType = desc.Type;
            mWidth = desc.Width;
            mHeight = desc.Height;
            mDepth = desc.Depth;

            if (desc.Format == Format.Unknown)
                throw new OptixException("OGLBuffer Error: Unknown buffer types not supported");

            if (desc.Format == Format.User && desc.ElemSize <= 0)
                throw new OptixException("D3DBuffer Error: ElemSize must be non-zero for User formats.");

            if (desc.Type == BufferType.InputOutputLocal)
                throw new OptixException("OGLBuffer Error: BufferType::InputOutputLocal Unsupported!");

            uint type = (uint)desc.Type;

            CheckError(GlInterop.rtBufferCreateFromGLBO(mContext.InternalPtr, type, desc.Resource, ref InternalPtr));

            CheckError(Api.rtBufferSetFormat(InternalPtr, (RTformat)desc.Format));

            if (desc.Format == Format.User)
            {
                CheckError(Api.rtBufferSetElementSize(InternalPtr, (uint)desc.ElemSize));
            }

            SetSize(desc.Width, desc.Height, desc.Depth);
        }

        /// <summary>
        /// Declares a OpenGl buffer as immutable and accessible by OptiX.
        /// Once registered, the buffer's properties will not be able to be modified
        /// </summary>
        public void Register()
        {
            CheckError(GlInterop.rtBufferGLRegister(InternalPtr));
        }

        public void Unregister()
        {
            CheckError(GlInterop.rtBufferGLUnregister(InternalPtr));

        }

    }
}