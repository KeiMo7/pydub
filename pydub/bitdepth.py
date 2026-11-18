"""
Explicit bit-depth conversion utilities.

The regular pydub pipeline keeps whatever sample width the input had
and converts between containers with audioop (plain truncation). This
module adds an opt-in, explicit path for high bit depth material
(e.g. 10-bit samples carried in 16-bit containers, the audio
equivalent of pixel formats like yuv420p10le) so that high precision
audio is never silently truncated to 8 bits.

Conventions used throughout this module:

- Samples are stored left-aligned (MSB first) inside their container,
  which matches audioop.lin2lin() scaling. A 10-bit sample value v is
  stored as ``v << 6`` in a 16-bit container.
- Nothing in here mutates global state, so all functions are safe to
  call concurrently from multiple threads.
- Log and exception messages never contain file paths, credentials or
  user data, only bit depths, formats and codec names.
"""
import array
import logging
import random
import wave

from io import BytesIO

from .exceptions import (
    UnsupportedBitDepthError,
    BitDepthMismatchError,
)
from .utils import (
    audioop,
    get_array_type,
)

logger = logging.getLogger("pydub.bitdepth")

# Logical bit depth -> byte width of the container samples are stored in.
CONTAINER_WIDTHS = {
    8: 1,
    10: 2,
    16: 2,
    24: 4,
    32: 4,
}

SUPPORTED_BIT_DEPTHS = tuple(sorted(CONTAINER_WIDTHS))

# PCM codec names matching each bit depth, used to validate explicit
# codec arguments instead of relying on encoder defaults.
PCM_CODECS = {
    8: 'pcm_u8',
    10: 'pcm_s16le',
    16: 'pcm_s16le',
    24: 'pcm_s24le',
    32: 'pcm_s32le',
}

# Containers the explicit path can write without lossy recompression.
PCM_FORMATS = ('wav', 'wave', 'raw')

DITHER_NONE = 'none'
DITHER_RECTANGULAR = 'rectangular'
DITHER_TRIANGULAR = 'triangular'
DITHER_STRATEGIES = (DITHER_NONE, DITHER_RECTANGULAR, DITHER_TRIANGULAR)

# Matches the historical behavior of plain truncation (audioop.lin2lin),
# so enabling the explicit path does not change existing output.
DEFAULT_DITHER = DITHER_NONE


def _tobytes(arr):
    try:
        return arr.tobytes()
    except AttributeError:
        return arr.tostring()


def check_bit_depth(bit_depth):
    if bit_depth not in CONTAINER_WIDTHS:
        raise UnsupportedBitDepthError(
            "bit depth %r is not supported; supported bit depths: %s" %
            (bit_depth, ", ".join(str(b) for b in SUPPORTED_BIT_DEPTHS)))


def check_dither(dither):
    if dither not in DITHER_STRATEGIES:
        raise ValueError(
            "unknown dither strategy %r; expected one of: %s" %
            (dither, ", ".join(DITHER_STRATEGIES)))


def validate_bit_depth(seg, declared_bit_depth=None):
    """
    Verify that the samples of an AudioSegment actually fit the declared
    bit depth. Raises BitDepthMismatchError when the declaration and the
    data disagree (e.g. 16-bit data declared as 8-bit).
    """
    declared = seg.bit_depth if declared_bit_depth is None else declared_bit_depth
    check_bit_depth(declared)

    width = seg.sample_width
    if CONTAINER_WIDTHS[declared] != width:
        raise BitDepthMismatchError(
            "declared %d-bit depth does not fit in %d-byte samples" %
            (declared, width))

    shift = width * 8 - declared
    if shift == 0 or not seg.raw_data:
        return True

    min_sample, max_sample = audioop.minmax(seg.raw_data, width)
    if (min_sample >> shift) < -(1 << (declared - 1)) or \
            (max_sample >> shift) > (1 << (declared - 1)) - 1:
        raise BitDepthMismatchError(
            "sample values exceed the range of the declared %d-bit depth" %
            declared)

    low_bits_mask = (1 << shift) - 1
    if any(sample & low_bits_mask for sample in seg.get_array_of_samples()):
        raise BitDepthMismatchError(
            "sample values use more precision than the declared %d-bit "
            "depth" % declared)

    return True


def convert_bit_depth(seg, target_bit_depth, dither=DEFAULT_DITHER):
    """
    Convert an AudioSegment to another bit depth through an explicit
    path. Upscaling (e.g. 8 -> 10 bit) is exact and introduces no
    artifacts; downscaling applies the requested dither strategy before
    quantization. The intermediate buffer always matches the target
    depth, no implicit truncation happens.
    """
    check_bit_depth(target_bit_depth)
    check_dither(dither)
    source_bit_depth = seg.bit_depth

    logger.debug("enter convert_bit_depth: %d -> %d bit (dither=%s)",
                 source_bit_depth, target_bit_depth, dither)
    try:
        validate_bit_depth(seg)
        if target_bit_depth == source_bit_depth:
            converted = seg
        elif target_bit_depth > source_bit_depth:
            converted = _upscale(seg, target_bit_depth)
        else:
            converted = _downscale(seg, target_bit_depth, dither)
    except Exception:
        logger.error("convert_bit_depth failed: %d -> %d bit",
                     source_bit_depth, target_bit_depth)
        raise
    logger.debug("exit convert_bit_depth: %d -> %d bit done",
                 source_bit_depth, target_bit_depth)
    return converted


def _upscale(seg, target_bit_depth):
    target_width = CONTAINER_WIDTHS[target_bit_depth]
    if target_width == seg.sample_width:
        # same container, more valid bits (e.g. 10 -> 16 bit): the
        # left-aligned data is already exact
        return seg._spawn(seg.raw_data,
                          overrides={'bit_depth': target_bit_depth})
    data = audioop.lin2lin(seg.raw_data, seg.sample_width, target_width)
    return seg._spawn(data, overrides={
        'sample_width': target_width,
        'frame_width': seg.channels * target_width,
        'bit_depth': target_bit_depth,
    })


def _downscale(seg, target_bit_depth, dither):
    target_width = CONTAINER_WIDTHS[target_bit_depth]
    overrides = {
        'sample_width': target_width,
        'frame_width': seg.channels * target_width,
        'bit_depth': target_bit_depth,
    }
    if dither == DITHER_NONE:
        # left-aligned samples make the container shift equal to the
        # logical bit reduction, so plain truncation is exact
        data = audioop.lin2lin(seg.raw_data, seg.sample_width, target_width)
        return seg._spawn(data, overrides=overrides)

    # quantization step in container units: the source container holds
    # (sample_width * 8 - target_bit_depth) bits below the target LSB
    shift = seg.sample_width * 8 - target_bit_depth
    pad = target_width * 8 - target_bit_depth
    min_val = -(1 << (target_bit_depth - 1))
    max_val = (1 << (target_bit_depth - 1)) - 1
    # a fresh Random instance per call keeps this thread-safe and does
    # not touch the global random state
    rng = random.Random()
    out = array.array(get_array_type(target_width * 8))
    append = out.append
    for sample in seg.get_array_of_samples():
        if dither == DITHER_RECTANGULAR:
            noise = rng.randrange(1 << shift)
        else:
            # zero-mean TPDF noise needs a half-LSB rounding offset to
            # stay unbiased under floor (arithmetic shift) quantization
            noise = (rng.randrange(1 << shift) - rng.randrange(1 << shift)
                     + (1 << (shift - 1)))
        value = (sample + noise) >> shift
        if value < min_val:
            value = min_val
        elif value > max_val:
            value = max_val
        append(value << pad)
    return seg._spawn(_tobytes(out), overrides=overrides)


def convert_batch(segments, target_bit_depth=None, dither=DEFAULT_DITHER):
    """
    Convert a batch of segments, deciding the bit depth per item. With
    no explicit target each item passes through at its own depth; the
    batch is never collapsed to the lowest common bit depth.
    """
    segments = list(segments)
    logger.debug("enter convert_batch: %d item(s), target=%s",
                 len(segments), target_bit_depth)
    try:
        converted = []
        for seg in segments:
            if target_bit_depth is None:
                validate_bit_depth(seg)
                converted.append(seg)
            else:
                converted.append(
                    convert_bit_depth(seg, target_bit_depth, dither=dither))
    except Exception:
        logger.error("convert_batch failed: %d item(s), target=%s",
                     len(segments), target_bit_depth)
        raise
    logger.debug("exit convert_batch: %d item(s) done", len(converted))
    return converted


def concatenate(segments, target_bit_depth=None, dither=DEFAULT_DITHER,
                crossfade=0):
    """
    Concatenate segments of mixed bit depth. Each segment is converted
    individually towards the target depth (default: the highest input
    depth, never the lowest) before being appended.
    """
    segments = list(segments)
    if not segments:
        raise ValueError("At least one AudioSegment instance is required")
    if target_bit_depth is None:
        target_bit_depth = max(seg.bit_depth for seg in segments)

    logger.debug("enter concatenate: %d segment(s), target=%d bit",
                 len(segments), target_bit_depth)
    try:
        converted = convert_batch(segments, target_bit_depth, dither=dither)
        channels = max(seg.channels for seg in converted)
        frame_rate = max(seg.frame_rate for seg in converted)
        converted = [
            seg.set_channels(channels).set_frame_rate(frame_rate)
            for seg in converted
        ]
        output = converted[0]
        for seg in converted[1:]:
            output = output.append(seg, crossfade=crossfade)
    except Exception:
        logger.error("concatenate failed: %d segment(s), target=%d bit",
                     len(segments), target_bit_depth)
        raise
    logger.debug("exit concatenate: %d segment(s) done", len(converted))
    return output


def export_with_bit_depth(seg, out_f, format, codec, bit_depth, dither):
    """
    Export an AudioSegment through the explicit bit-depth path. Only PCM
    containers are supported; anything else raises
    UnsupportedBitDepthError with a suggestion instead of silently
    falling back to 8 bits. The wav payload is assembled in memory first
    so a failure cannot leave a half-written file with a mismatched bit
    depth behind.
    """
    logger.debug("enter explicit export: %d bit, format=%s",
                 bit_depth, format)
    try:
        check_bit_depth(bit_depth)
        if dither is None:
            dither = DEFAULT_DITHER
        check_dither(dither)

        fmt = format.lower()
        if fmt not in PCM_FORMATS:
            raise UnsupportedBitDepthError(
                "format %r cannot carry %d-bit PCM samples; use a PCM "
                "container such as 'wav' or 'raw', or export without "
                "bit_depth to use the encoder defaults" %
                (format, bit_depth))

        expected_codec = PCM_CODECS[bit_depth]
        if codec is not None and codec != expected_codec:
            raise UnsupportedBitDepthError(
                "codec %r cannot carry %d-bit PCM samples; use %r or "
                "omit the codec argument" %
                (codec, bit_depth, expected_codec))

        converted = convert_bit_depth(seg, bit_depth, dither=dither)

        if fmt == 'raw':
            out_f.write(converted.raw_data)
            out_f.seek(0)
            logger.debug("exit explicit export: %d bit raw done", bit_depth)
            return out_f

        sampwidth, pcm_data = _pack_for_wave(converted)
        buffer = BytesIO()
        wave_data = wave.open(buffer, 'wb')
        wave_data.setnchannels(converted.channels)
        wave_data.setsampwidth(sampwidth)
        wave_data.setframerate(converted.frame_rate)
        wave_data.setnframes(int(converted.frame_count()))
        wave_data.writeframesraw(pcm_data)
        wave_data.close()

        out_f.write(buffer.getvalue())
        out_f.seek(0)
    except Exception:
        logger.error("explicit export failed: %d bit, format=%s",
                     bit_depth, format)
        raise
    logger.debug("exit explicit export: %d bit wav done", bit_depth)
    return out_f


def _pack_for_wave(seg):
    """
    Return (sampwidth, pcm_bytes) for the wave module. 24-bit samples
    are carried in 32-bit containers internally, so the low padding byte
    is dropped explicitly instead of relying on encoder defaults.
    """
    if seg.bit_depth == 24:
        raw = seg.raw_data
        pcm_data = b''.join(
            raw[pos + 1:pos + 4] for pos in range(0, len(raw), 4))
        return 3, pcm_data
    if seg.bit_depth == 8:
        # wav stores 8-bit samples unsigned
        return 1, audioop.bias(seg.raw_data, 1, 128)
    return seg.sample_width, seg.raw_data
