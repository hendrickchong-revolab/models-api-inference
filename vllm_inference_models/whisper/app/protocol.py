from typing import Literal, Optional

from pydantic import Field
from typing_extensions import TypeAlias
from vllm.entrypoints.openai.protocol import TranscriptionRequest

ChunkingMethod: TypeAlias = Literal["naive", "vad"]


class TranscriptionRequestImprove(TranscriptionRequest):
    client_reference_id: Optional[str] = None
    """
    - Client Reference ID which will be used as Request ID
    """
    chunking_method: ChunkingMethod = Field(default="vad")
    """
    - For `naive`, will chunk the audio every 30s regardless voice activities.
    - For `vad`, will use Silero VAD to chunk the audio based on parameters below.
    """
    minimum_silent_ms: int = 1000
    """
    Minimum silents accumulated to trigger transcription.
    A positive chunk equal to minimum_silent_ms + minimum_trigger_vad_ms
    """
    minimum_trigger_vad_ms: int = 2000
    """
    Minimum voice activities to trigger transcription.
    A positive chunk equal to minimum_silent_ms + minimum_trigger_vad_ms
    """
    reject_segment_vad_ratio: float = 0.9
    """
    If the segments is less percent than reject threshold, it will trigger transcription.
    """
    vad_confidence: float = 0.5
    """
    Confidence probability to accept a segment is a positive voice activity.
    """
