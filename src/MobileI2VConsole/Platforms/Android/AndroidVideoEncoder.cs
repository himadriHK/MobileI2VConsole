using Android.Media;
using Android.OS;
using Java.Nio;
using MobileI2VConsole.Services;

namespace MobileI2VConsole.Platforms.Android;

/// <summary>
/// Android-specific video encoder that wraps MediaCodec to encode raw RGBA frames
/// into an H.264 .mp4 video file.
/// </summary>
public class AndroidVideoEncoder : IVideoEncoder
{
    // TI-specific flexible YUV420 semi-planar color format constant (0x7F000100)
    private const int ColorFormatYuv420 = 2130706688;

    public async Task<string> EncodeFramesAsync(
        byte[][] frames,
        string outputPath,
        int width,
        int height,
        int fps,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        if (frames == null || frames.Length == 0)
            throw new ArgumentException("At least one frame is required");

        // Ensure output directory exists
        var outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir))
            Directory.CreateDirectory(outputDir);

        // Remove existing file if present
        if (File.Exists(outputPath))
            File.Delete(outputPath);

        var mimeType = MediaFormat.MimetypeVideoAvc;
        const int bitRate = 4_000_000; // 4 Mbps
        const int iFrameInterval = 1; // 1 second

        // Configure MediaFormat
        var format = MediaFormat.CreateVideoFormat(mimeType, width, height);
        format.SetInteger(MediaFormat.KeyColorFormat, ColorFormatYuv420);
        format.SetInteger(MediaFormat.KeyBitRate, bitRate);
        format.SetInteger(MediaFormat.KeyFrameRate, fps);
        format.SetInteger(MediaFormat.KeyIFrameInterval, iFrameInterval);

        using var mediaCodec = MediaCodec.CreateEncoderByType(mimeType);
        if (mediaCodec == null)
            throw new InvalidOperationException("Failed to create MediaCodec encoder for H.264");

        mediaCodec.Configure(format, null, null, MediaCodecConfigFlags.Encode);
        mediaCodec.Start();

        var outputFile = new Java.IO.File(outputPath);
        var pfd = ParcelFileDescriptor.Open(outputFile, ParcelFileMode.ReadWrite);
        using var mediaMuxer = new MediaMuxer(pfd.FileDescriptor, 0); // 0 = OutputFormat.Mpeg4

        var trackIndex = -1;
        var muxerStarted = false;
        var frameCount = 0;
        var totalFrames = frames.Length;
        var presentationTimeUs = 0L;
        var frameDurationUs = 1_000_000L / fps;

        var inputBuffers = mediaCodec.GetInputBuffers();
        var outputBuffers = mediaCodec.GetOutputBuffers();

        var inputFrameIndex = 0;
        var outputDone = false;

        while (!outputDone && !ct.IsCancellationRequested)
        {
            // Feed input frames
            if (inputFrameIndex < totalFrames)
            {
                var inputBufferIndex = mediaCodec.DequeueInputBuffer(10_000); // 10ms timeout
                if (inputBufferIndex >= 0)
                {
                    var inputBuffer = inputBuffers[inputBufferIndex];
                    inputBuffer.Clear();

                    // Convert RGBA to NV12 (YUV420 semi-planar)
                    var yuvData = ConvertRgbaToNv12(frames[inputFrameIndex], width, height);
                    inputBuffer.Put(yuvData);

                    var flags = inputFrameIndex == totalFrames - 1
                        ? MediaCodecBufferFlags.EndOfStream
                        : 0;

                    mediaCodec.QueueInputBuffer(inputBufferIndex, 0, yuvData.Length,
                        presentationTimeUs, flags);

                    inputFrameIndex++;
                    presentationTimeUs += frameDurationUs;
                }
            }

            // Dequeue output
            var outputBufferInfo = new MediaCodec.BufferInfo();
            var outputBufferIndex = mediaCodec.DequeueOutputBuffer(outputBufferInfo, 10_000);

            if (outputBufferIndex >= 0)
            {
                var outputBuffer = outputBuffers[outputBufferIndex];
                var isConfigFrame = (outputBufferInfo.Flags & MediaCodecBufferFlags.CodecConfig) != 0;

                if (!isConfigFrame && outputBufferInfo.Size > 0)
                {
                    if (!muxerStarted)
                    {
                        trackIndex = mediaMuxer.AddTrack(mediaCodec.OutputFormat);
                        mediaMuxer.Start();
                        muxerStarted = true;
                    }

                    outputBuffer.Position(outputBufferInfo.Offset);
                    var encodedData = new byte[outputBufferInfo.Size];
                    outputBuffer.Get(encodedData, 0, outputBufferInfo.Size);

                    mediaMuxer.WriteSampleData(trackIndex, ByteBuffer.Wrap(encodedData),
                        outputBufferInfo);

                    frameCount++;
                    progress?.Report((double)frameCount / totalFrames);
                }

                mediaCodec.ReleaseOutputBuffer(outputBufferIndex, false);

                if ((outputBufferInfo.Flags & MediaCodecBufferFlags.EndOfStream) != 0)
                    outputDone = true;
            }
            else if (outputBufferIndex == (int)MediaCodecInfoState.OutputBuffersChanged)
            {
                outputBuffers = mediaCodec.GetOutputBuffers();
            }
            else if (outputBufferIndex == (int)MediaCodecInfoState.TryAgainLater)
            {
                if (inputFrameIndex >= totalFrames)
                {
                    // No more input, wait briefly for remaining output
                    await Task.Delay(10, ct);
                }
            }
        }

        // Flush and stop
        try { mediaCodec.SignalEndOfInputStream(); } catch { }
        mediaCodec.Stop();
        mediaCodec.Release();

        if (muxerStarted)
        {
            try { mediaMuxer.Stop(); } catch { }
        }
        mediaMuxer.Release();

        ct.ThrowIfCancellationRequested();

        if (!File.Exists(outputPath))
            throw new InvalidOperationException("Video encoding failed — output file not created");

        return outputPath;
    }

    /// <summary>
    /// Converts RGBA byte array to NV12 (YUV420 semi-planar) format
    /// required by Android MediaCodec.
    /// </summary>
    private static byte[] ConvertRgbaToNv12(byte[] rgba, int width, int height)
    {
        int frameSize = width * height;
        int ySize = frameSize;
        int uvSize = frameSize / 2; // UV plane is half the size (quarter resolution × 2 channels)
        var nv12 = new byte[ySize + uvSize];

        int yIndex = 0;
        int uvIndex = ySize;

        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int pixelIndex = (j * width + i) * 4;
                byte r = rgba[pixelIndex];
                byte g = rgba[pixelIndex + 1];
                byte b = rgba[pixelIndex + 2];
                // a = rgba[pixelIndex + 3]; // alpha unused for video

                // Y = 0.299*R + 0.587*G + 0.114*B
                int y = (r * 77 + g * 150 + b * 29) >> 8;
                nv12[yIndex++] = (byte)Math.Clamp(y, 0, 255);

                // U/V at half resolution
                if (j % 2 == 0 && i % 2 == 0)
                {
                    // U = -0.169*R - 0.331*G + 0.499*B + 128
                    int u = ((-r * 43 - g * 84 + b * 127) >> 8) + 128;
                    // V = 0.499*R - 0.418*G - 0.0813*B + 128
                    int v = ((r * 127 - g * 107 - b * 21) >> 8) + 128;

                    nv12[uvIndex++] = (byte)Math.Clamp(v, 0, 255); // V first in NV12
                    nv12[uvIndex++] = (byte)Math.Clamp(u, 0, 255); // U second
                }
            }
        }

        return nv12;
    }
}
