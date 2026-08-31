###############################################################################
#  Loopback-oriented WebRTC encoder bitrate tuning
###############################################################################
"""Raise aiortc's software-encoder bitrates for same-machine playback.

aiortc sizes its encoders for real networks: H264 starts at 1 Mbps and is
clamped to 3 Mbps, VP8 to 1.5 Mbps.  Xiaoman serves a browser on the same
host, so the only cost of a higher bitrate is CPU — libx264 measures at about
3 ms/frame for 704x896 at 8 Mbps, well inside the 40 ms frame budget.

Measured on an M4 Max against this project's own avatar frames, encoding
60 real frames and comparing the decoded result to the source:

    1 Mbps -> 39.67 dB    3 Mbps -> 42.08 dB    8 Mbps -> 42.93 dB

So 1 -> 3 Mbps is worth 2.4 dB while 3 -> 8 Mbps buys only another 0.85 dB.
Raising the *default* matters more than raising the ceiling, because aiortc
starts every session at the default and only climbs as REMB feedback arrives.

The constants live at module scope in aiortc and are read at call time — the
encoder reads ``DEFAULT_BITRATE`` in ``__init__`` and the ``target_bitrate``
setter re-reads ``MIN``/``MAX`` on every REMB update — so rebinding them
before the first peer connection is sufficient and needs no aiortc fork.
"""

from __future__ import annotations

from aiortc.codecs import h264 as aiortc_h264
from aiortc.codecs import vpx as aiortc_vpx

from utils.logger import logger

# aiortc only offers profile-level-id 42001f / 42e01f, i.e. Constrained
# Baseline Level 3.1, whose MaxBR is 14000 kbps.  Encoding above the level we
# advertised in SDP risks a decoder that trusts the negotiated ceiling.
H264_LEVEL_31_MAX_BITRATE = 14_000_000


def apply_webrtc_bitrate(default_bitrate: int, max_bitrate: int) -> dict[str, int]:
    """Rebind aiortc's encoder bitrate ceilings; return what was applied."""

    default_bitrate = int(default_bitrate)
    max_bitrate = int(max_bitrate)
    if default_bitrate <= 0 or max_bitrate <= 0:
        raise ValueError("video bitrates must be positive")
    if default_bitrate > max_bitrate:
        raise ValueError(
            f"video_bitrate {default_bitrate} exceeds video_max_bitrate {max_bitrate}"
        )

    if max_bitrate > H264_LEVEL_31_MAX_BITRATE:
        logger.warning(
            "video_max_bitrate %d exceeds the advertised H264 Level 3.1 ceiling; "
            "clamping to %d",
            max_bitrate,
            H264_LEVEL_31_MAX_BITRATE,
        )
        max_bitrate = H264_LEVEL_31_MAX_BITRATE
        default_bitrate = min(default_bitrate, max_bitrate)

    aiortc_h264.DEFAULT_BITRATE = default_bitrate
    aiortc_h264.MAX_BITRATE = max_bitrate
    # VP8 is only reached when a browser refuses H264, but leaving it at
    # 1.5 Mbps would make that fallback look markedly worse than the primary
    # path for no reason.
    aiortc_vpx.DEFAULT_BITRATE = default_bitrate
    aiortc_vpx.MAX_BITRATE = max_bitrate

    applied = {"default_bitrate": default_bitrate, "max_bitrate": max_bitrate}
    logger.info(
        "webrtc video bitrate: default=%.1f Mbps max=%.1f Mbps",
        default_bitrate / 1e6,
        max_bitrate / 1e6,
    )
    return applied
