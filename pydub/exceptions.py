class PydubException(Exception):
    """
    Base class for any Pydub exception
    """


class TooManyMissingFrames(PydubException):
    pass


class InvalidDuration(PydubException):
    pass


class InvalidTag(PydubException):
    pass


class InvalidID3TagVersion(PydubException):
    pass


class CouldntDecodeError(PydubException):
    pass


class CouldntEncodeError(PydubException):
    pass


class MissingAudioParameter(PydubException):
    pass


class UnsupportedBitDepthError(PydubException):
    """
    Raised when a bit depth, container or codec cannot carry the
    requested sample precision. Never falls back to a lower bit
    depth implicitly.
    """


class BitDepthMismatchError(PydubException):
    """
    Raised when the declared bit depth of an AudioSegment does not
    match the actual sample values (e.g. samples exceed the range
    of the declared bit depth).
    """
