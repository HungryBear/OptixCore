using System;
using OptixCore.Library.Native;

namespace OptixCore.Library
{
    public enum WrapMode
    {
        /// <summary>
        /// Texture coordinates are wrapped back around to the [0,1] range when they go outside the [0,1] range.
        /// </summary>
        Repeat = RTwrapmode.RT_WRAP_REPEAT,

        /// <summary>
        /// Texture coordinates are clamped to the [0,1] range.
        /// </summary>
        Clamp = RTwrapmode.RT_WRAP_CLAMP_TO_EDGE
    };

    /// <summary>
    /// Controls the filtering mode for texture fetches
    /// </summary>
    public enum FilterMode
    {
        /// <summary>
        /// No texture filtering.
        /// </summary>
        None = RTfiltermode.RT_FILTER_NONE,

        /// <summary>
        /// Finds the closest texture sample to the texture coordinate.
        /// </summary>
        Nearest = RTfiltermode.RT_FILTER_NEAREST,

        /// <summary>
        /// Performs a bilinear or bicubic interpolation of the texture sample for 2D and 3D textures respectively.
        /// </summary>
        Linear = RTfiltermode.RT_FILTER_LINEAR
    };

    /// <summary>
    /// Controls the interpretation of texture samples. 
    /// </summary>
    public enum TextureReadMode
    {
        /// <summary>
        /// Data returned by a sample will be in the original ElementType range
        /// </summary>
        ElementType = RTtexturereadmode.RT_TEXTURE_READ_ELEMENT_TYPE,

        /// <summary>
        /// Data returned by a sample will be in a normalized [0,1] range
        /// </summary>
        NormalizedFloat = RTtexturereadmode.RT_TEXTURE_READ_NORMALIZED_FLOAT
    };

    /// <summary>
    /// Controls the interpretation of texture coordinates. 
    /// </summary>
    public enum TextureIndexMode
    {
        /// <summary>
        /// NormalizeCoords will parameterize the texture over [0,1].
        /// </summary>
        NormalizeCoords = RTtextureindexmode.RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,

        /// <summary>
        /// ArrayIndex will interpret texture coordinates as array indices [0, N] into the underlying buffer
        /// </summary>
        ArrayIndex = RTtextureindexmode.RT_TEXTURE_INDEX_ARRAY_INDEX
    };

    /// <summary>
    /// Holds <see cref="WrapMode">WrapModes</see> for U, V, and W texture dimensions.
    /// </summary>
    public struct WrapUVW
    {
        public WrapMode WrapU;
        public WrapMode WrapV;
        public WrapMode WrapW;

        public static WrapUVW Clamp
        {
            get
            {
                WrapUVW wrap = new WrapUVW(WrapMode.Clamp, WrapMode.Clamp, WrapMode.Clamp);
                return wrap;
            }
        }

        public static WrapUVW Repeat
        {
            get
            {
                WrapUVW wrap = new WrapUVW(WrapMode.Repeat, WrapMode.Repeat, WrapMode.Repeat);
                return wrap;
            }
        }

        public WrapUVW(WrapMode u, WrapMode v, WrapMode w)
        {
            WrapU = u;
            WrapV = v;
            WrapW = w;
        }
    }

    /// <summary>
    /// Holds <see cref="FilterMode">FilterModes</see> for Minification, Magnification, and Mipmaps.
    /// </summary>
    public struct FilterMinMagMip
    {
        public FilterMode Min;
        public FilterMode Mag;
        public FilterMode Mip;
    };

    /// <summary>
    /// Description data used to create a <see cref="TextureSampler">TextureSampler</see>.
    /// </summary>
    public struct TextureSamplerDesc
    {
        /// <summary>
        /// <see cref="WrapUVW">Wrap modes</see> for the TextureSampler.
        /// </summary>
        public WrapUVW Wrap;

        /// <summary>
        /// <see cref="FilterMinMagMip">Filter modes</see> for the TextureSampler.
        /// </summary>
        public FilterMinMagMip Filter;

        /// <summary>
        /// <see cref="TextureReadMode">TextureReadMode</see> for the TextureSampler.<br/>
        /// For ElementType, data returned by a sample will be in the original ElementType range<br/>
        /// For NormalizedFloat, data returned by a sample will be in a normalized [0,1] range
        /// </summary>
        public TextureReadMode Read;

        /// <summary>
        /// <see cref="TextureIndexMode">Texture indexing mode</see> for the TextureSampler.
        /// Controls the interpretation of texture coordinates. 
        /// NormalizeCoords will parameterize the texture over [0,1].
        /// ArrayIndex will interpret texture coordinates as array indices into the underlying buffer
        /// </summary>
        public TextureIndexMode Index;

        /// <summary>
        /// Number of mip map levels for the sampler
        /// </summary>
        public uint MipLevels;

        /// <summary>
        /// Max anisotropy value for anisotropic filtering. A value > 0.0f will enable anisotropic filtering.
        /// </summary>
        public float MaxAnisotropy;

        public TextureSamplerDesc(WrapUVW wrap, FilterMinMagMip filter, TextureReadMode normalizedFloat, TextureIndexMode normalizeCoords, uint v1, float v2) : this()
        {
            Wrap = wrap;
            Filter = filter;
            Read = normalizedFloat;
            Index = normalizeCoords;
            MipLevels = v1;
            MaxAnisotropy = v2;
        }

        public static TextureSamplerDesc GetDefault(WrapMode wrapmode)
        {
            var wrap = new WrapUVW(wrapmode, wrapmode, wrapmode);

            TextureSamplerDesc desc = Default;
            desc.Wrap = wrap;

            return desc;
        }

        /// <summary>
        /// Gets a default TextureSamplerDesc with settings:
        /// WrapUVW = [ Clamp, Clamp, Clamp ]
        /// Filter = [ Linear, Linear, Linear ]
        /// Read = NormalizedFloat
        /// Index = NormalizeCoords
        /// MipLevels = 1
        /// MaxAnisotropy = 0.0f
        /// </summary>
        public static TextureSamplerDesc Default
        {
            get
            {
                var wrap = new WrapUVW(WrapMode.Clamp, WrapMode.Clamp, WrapMode.Clamp);
                var filter = new FilterMinMagMip { Min = FilterMode.Linear, Mag = FilterMode.Linear, Mip = FilterMode.None };
                var desc = new TextureSamplerDesc(wrap, filter, TextureReadMode.NormalizedFloat, TextureIndexMode.NormalizeCoords, 1, 0.0f);
                return desc;
            }
        }
    }

    public class TextureSampler : DataNode
    {

        /// <summary>
        /// Create a TextureSampler from a <see cref="TextureSamplerDesc">TextureSamplerDesc</see>.
        /// </summary>
        /// <param name="context">Optix context</param>
        /// <param name="desc">Description of the TextureSampler</param>
        public TextureSampler(Context context, TextureSamplerDesc desc) : base(context)
        {
            CheckError(Api.rtTextureSamplerCreate(context.InternalPtr, out InternalPtr));
            SetDesc(ref desc);
        }

        public TextureSampler(Context ctx) : base(ctx) { }

        private void SetDesc(ref TextureSamplerDesc desc)
        {
            CheckError(Api.rtTextureSamplerSetWrapMode(InternalPtr, 0, (RTwrapmode)desc.Wrap.WrapU));
            CheckError(Api.rtTextureSamplerSetWrapMode(InternalPtr, 1, (RTwrapmode)desc.Wrap.WrapV));
            CheckError(Api.rtTextureSamplerSetWrapMode(InternalPtr, 2, (RTwrapmode)desc.Wrap.WrapW));
            CheckError(Api.rtTextureSamplerSetFilteringModes(InternalPtr, (RTfiltermode)desc.Filter.Min, (RTfiltermode)desc.Filter.Mag, (RTfiltermode)desc.Filter.Mip));
            CheckError(Api.rtTextureSamplerSetReadMode(InternalPtr, (RTtexturereadmode)desc.Read));
            CheckError(Api.rtTextureSamplerSetIndexingMode(InternalPtr, (RTtextureindexmode)desc.Index));
            CheckError(Api.rtTextureSamplerSetMaxAnisotropy(InternalPtr, desc.MaxAnisotropy));
        }
        public int GetId()
        {
            int id = -1;
            CheckError(Api.rtTextureSamplerGetId(InternalPtr, ref id));
            return id;
        }

        public virtual void SetBuffer(OptixBuffer buffer)
        {
            CheckError(Api.rtTextureSamplerSetBuffer(InternalPtr, 0u, 0u, buffer.InternalPtr));
        }
        /*
         
		/// <summary>
		/// Gets the Texture addressing mode for U, V, W coordinates
        /// </summary>
		property WrapUVW Wrap
		{
			WrapUVW get()
			{
				RTwrapmode u, v, w;

				CheckError( rtTextureSamplerGetWrapMode( InternalPtr, 0, &u ) );
				CheckError( rtTextureSamplerGetWrapMode( InternalPtr, 1, &v ) );
				CheckError( rtTextureSamplerGetWrapMode( InternalPtr, 2, &w ) );

				WrapUVW mode = { static_cast<WrapMode>( u ), static_cast<WrapMode>( v ), static_cast<WrapMode>( w ) };
				return mode;
			}
		}

		/// <summary>
		/// Gets the Texture filtering mode for the sampler
        /// </summary>
		property FilterMinMagMip Filter
		{
			FilterMinMagMip get()
			{
				RTfiltermode min, mag, mip;
				CheckError( rtTextureSamplerGetFilteringModes( InternalPtr, &min, &mag, &mip ) );

				FilterMinMagMip mode = { static_cast<FilterMode>( min ), static_cast<FilterMode>( mag ), static_cast<FilterMode>( mip ) };
				return mode;
			}
		}

		/// <summary>
		/// Gets the texture read mode for the sampler.
        /// </summary>
		property TextureReadMode ReadMode
		{
			TextureReadMode get()
			{
				RTtexturereadmode mode;
				CheckError( rtTextureSamplerGetReadMode( InternalPtr, &mode ) );

				return static_cast<TextureReadMode>( mode );
			}
		}

		/// <summary>
		/// Gets the texture indexing mode for the sampler.
        /// </summary>
		property TextureIndexMode IndexingMode
		{
			TextureIndexMode get()
			{
				RTtextureindexmode mode;
				CheckError( rtTextureSamplerGetIndexingMode( InternalPtr, &mode ) );

				return static_cast<TextureIndexMode>( mode );
			}
		}

		/// <summary>
		/// Gets the number of mip map levels for the sampler
        /// </summary>
		property unsigned int MipLevelCount
		{
			unsigned int get()
			{
				unsigned int count;
				CheckError( rtTextureSamplerGetMipLevelCount( InternalPtr, &count ) );

				return count;
			}
		}
         */

        /// <summary>
        /// Gets the max anisotropy
        /// </summary>
        public float MaxAnisotropy
        {
            get
            {
                float ani=0f;
                CheckError(Api.rtTextureSamplerGetMaxAnisotropy(InternalPtr, ref ani));

                return ani;
            }
        }
        public override void Validate()
        {
            CheckError(Api.rtTextureSamplerValidate(InternalPtr));

        }

        public override void Destroy()
        {
            if (InternalPtr != IntPtr.Zero)
                CheckError(Api.rtTextureSamplerDestroy(InternalPtr));

            InternalPtr = IntPtr.Zero;
        }

        public override IntPtr ObjectPtr()
        {
            return InternalPtr;
        }
    }
}