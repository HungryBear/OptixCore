using System;
using System.Collections.Generic;
using System.Text;
using OptixCore.Library;

namespace Playground
{
    public class DenoiserWindow : OptixWindow
    {
        bool denoiser_perf_mode = false;
        int denoiser_perf_iter = 1;

        // Mouse state
        Int2 mouse_prev_pos;
        int mouse_button;

        // Post-processing
        OptixCommandList commandListWithDenoiser;
        OptixCommandList commandListWithoutDenoiser;
        OptixPostprocessingStage tonemapStage;
        OptixPostprocessingStage denoiserStage;
        OptixBuffer denoisedBuffer;
        OptixBuffer emptyBuffer;
        OptixBuffer trainingDataBuffer;


        // number of frames that show the original image before switching on denoising
        int numNonDenoisedFrames = 4;

        // Defines the amount of the original image that is blended with the denoised result
        // ranging from 0.0 to 1.0
        float denoiseBlend = 0f;

        // Defines which buffer to show.
        // 0 - denoised 1 - original, 2 - tonemapped, 3 - albedo, 4 - normal
        int showBuffer = 0;

        // The denoiser mode.
        // 0 - RGB only, 1 - RGB + albedo, 2 - RGB + albedo + normals
        int denoiseMode = 0;


        // The path to the training data file set with -t or empty
        string training_file;

        // The path to the second training data file set with -t2 or empty
        string training_file_2;

        // Toggles between using custom training data (if set) or the built in training data.
        bool useCustomTrainingData = true;

        // Toggles the custom data between the one specified with -t1 and -t2, if available.
        bool useFirstTrainingDataPath = true;

        // Contains info for the currently shown buffer
        string bufferInfo;

        public DenoiserWindow() : base(512, 512)
        {
            UsePBO = true;
        }

        protected override void Initialize()
        {
            base.Initialize();
            OptixContext = new Context();
            OptixContext.RayTypeCount = 2;
            OptixContext.EntryPointCount = 1;
            OptixContext.EnableAllExceptions = false;
            OptixContext.SetStackSize(1800);

            OptixContext["scene_epsilon"].Set(1e-3f);
            const int rr_begin_depth = 1;
            OptixContext["rr_begin_depth"].Set(rr_begin_depth);

            //var renderBuffer = CreateOutputBuffer()

            //OptixContext.SetUsageReportCallback
            /*
              Buffer renderBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(renderBuffer);
    Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["tonemapped_buffer"]->set(tonemappedBuffer); 
    Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_albedo_buffer"]->set(albedoBuffer);

    // The normal buffer use float4 for performance reasons, the fourth channel will be ignored.
    Buffer normalBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_normal_buffer"]->set(normalBuffer);

    denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);
    trainingDataBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

    // Setup programs
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "optixDenoiser.cu" );
    context->setRayGenerationProgram( 0, context->createProgramFromPTXString( ptx, "pathtrace_camera" ) );
    context->setExceptionProgram( 0, context->createProgramFromPTXString( ptx, "exception" ) );
    context->setMissProgram( 0, context->createProgramFromPTXString( ptx, "miss" ) );

    context[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
    context[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
    context[ "bg_color"         ]->setFloat( make_float3(0.0f) );

             
             */

        }
    }
}
