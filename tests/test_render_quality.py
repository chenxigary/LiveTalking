"""Tests for the render-quality changes: paste-back, stride, encoder bitrate.

Run from the LiveTalking root with the Avatar environment:

    ../../../.venv-v3-avatar/bin/python -m unittest discover -s tests
"""

import unittest
from types import SimpleNamespace

import numpy as np

from avatars.base_avatar import BaseAvatar
from server.codec_tuning import H264_LEVEL_31_MAX_BITRATE, apply_webrtc_bitrate
from utils.image import _lower_face_alpha, blend_lower_face_bgr


class LowerFaceAlphaTests(unittest.TestCase):
    def test_mask_is_zero_above_the_split_and_one_in_the_interior(self):
        alpha = _lower_face_alpha(100, 80, 0.5, 10, 4)
        self.assertTrue(np.all(alpha[:50] == 0.0))
        # Past the feather ramp and inside the edge fade the patch wins fully.
        self.assertEqual(alpha[70, 40, 0], 1.0)

    def test_feather_ramps_monotonically_from_the_split(self):
        alpha = _lower_face_alpha(100, 80, 0.5, 10, 0)
        column = alpha[50:60, 40, 0]
        self.assertEqual(column[0], 0.0)
        self.assertTrue(np.all(np.diff(column) > 0))
        self.assertEqual(alpha[60, 40, 0], 1.0)

    def test_edge_fade_reaches_zero_at_the_crop_border(self):
        alpha = _lower_face_alpha(100, 80, 0.5, 0, 6)
        self.assertEqual(alpha[70, 0, 0], 0.0)
        self.assertEqual(alpha[70, 79, 0], 0.0)
        self.assertEqual(alpha[99, 40, 0], 0.0)

    def test_oversized_edge_is_clamped_instead_of_wrapping(self):
        alpha = _lower_face_alpha(20, 20, 0.5, 0, 999)
        self.assertEqual(alpha.shape, (20, 20, 1))
        self.assertTrue(np.all(alpha >= 0.0))
        self.assertTrue(np.all(alpha <= 1.0))

    def test_cached_mask_is_not_writable_by_callers(self):
        alpha = _lower_face_alpha(40, 40, 0.5, 4, 2)
        with self.assertRaises(ValueError):
            alpha[0, 0, 0] = 1.0


class BlendLowerFaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260830)
        self.canvas = rng.integers(0, 256, (120, 100, 3), dtype=np.uint8)
        self.patch = np.full((64, 64, 3), 255, dtype=np.uint8)
        self.box = (20, 100, 10, 70)  # y1, y2, x1, x2

    def test_pixels_above_the_blend_line_are_left_untouched(self):
        y1, y2, x1, x2 = self.box
        before = self.canvas.copy()
        result = blend_lower_face_bgr(self.canvas, self.patch, self.box)
        mid = (y2 - y1) // 2
        np.testing.assert_array_equal(result[: y1 + mid], before[: y1 + mid])
        np.testing.assert_array_equal(result[y1:y1 + mid, x1:x2],
                                      before[y1:y1 + mid, x1:x2])

    def test_everything_outside_the_face_box_is_left_untouched(self):
        y1, y2, x1, x2 = self.box
        before = self.canvas.copy()
        blend_lower_face_bgr(self.canvas, self.patch, self.box)
        np.testing.assert_array_equal(self.canvas[:, :x1], before[:, :x1])
        np.testing.assert_array_equal(self.canvas[:, x2:], before[:, x2:])
        np.testing.assert_array_equal(self.canvas[y2:], before[y2:])

    def test_interior_of_the_lower_face_takes_the_patch(self):
        blend_lower_face_bgr(self.canvas, self.patch, self.box,
                             feather=4, edge=4)
        # Well inside both the feather ramp and the edge fade.
        self.assertTrue(np.all(self.canvas[80, 40] == 255))

    def test_blend_is_in_place_and_returns_the_same_array(self):
        result = blend_lower_face_bgr(self.canvas, self.patch, self.box)
        self.assertIs(result, self.canvas)

    def test_degenerate_box_is_a_no_op_rather_than_an_exception(self):
        before = self.canvas.copy()
        result = blend_lower_face_bgr(self.canvas, self.patch, (30, 30, 10, 70))
        np.testing.assert_array_equal(result, before)


class _StrideHarness:
    """Minimal stand-in that exercises BaseAvatar's stride controller alone.

    Constructing a real BaseAvatar would pull in a TTS plugin, an output
    transport and the custom-video loader, none of which the control law
    touches.
    """

    observe = BaseAvatar._observe_inference

    def __init__(self, limit: int, mode: str = "adaptive", fps: int = 25):
        self.inference_stride = limit
        self.inference_stride_mode = mode
        self._active_stride = 1 if mode == "adaptive" else limit
        self._inference_ema_sec = 0.0
        self._inference_samples = 0
        self.opt = SimpleNamespace(fps=fps)

    def warm(self):
        """Consume the discarded cold-start sample."""
        self.observe(0.0136)
        return self


class AdaptiveStrideTests(unittest.TestCase):
    def test_measured_m4_max_latency_keeps_every_frame_inferred(self):
        harness = _StrideHarness(4)
        for _ in range(10):
            harness.observe(0.0136)  # 13.6ms, the measured batch=1 figure
        self.assertEqual(harness._active_stride, 1)

    def test_cold_start_compile_does_not_widen_the_stride(self):
        harness = _StrideHarness(4)
        harness.observe(1.122)  # measured first-inference Metal compile
        self.assertEqual(harness._active_stride, 1)
        self.assertEqual(harness._inference_ema_sec, 0.0)

    def test_saturated_gpu_widens_the_stride_up_to_the_limit(self):
        harness = _StrideHarness(4).warm()
        harness.observe(0.1155)  # 115ms, measured under a saturating MPS load
        self.assertEqual(harness._active_stride, 4)

    def test_stride_never_exceeds_the_configured_limit(self):
        harness = _StrideHarness(2).warm()
        harness.observe(1.0)
        self.assertEqual(harness._active_stride, 2)

    def test_intermediate_latency_picks_an_intermediate_stride(self):
        harness = _StrideHarness(4).warm()
        harness.observe(0.050)  # ceil(50ms * 1.25 / 40ms) == 2
        self.assertEqual(harness._active_stride, 2)

    def test_stride_recovers_once_the_gpu_frees_up(self):
        harness = _StrideHarness(4).warm()
        harness.observe(0.1155)
        self.assertEqual(harness._active_stride, 4)
        for _ in range(20):
            harness.observe(0.0136)
        self.assertEqual(harness._active_stride, 1)

    def test_fixed_mode_ignores_measured_latency(self):
        harness = _StrideHarness(4, mode="fixed")
        harness.observe(0.0001)
        self.assertEqual(harness._active_stride, 4)
        harness.observe(5.0)
        self.assertEqual(harness._active_stride, 4)

    def test_fixed_mode_still_records_latency_for_telemetry(self):
        harness = _StrideHarness(4, mode="fixed").warm()
        harness.observe(0.020)
        self.assertAlmostEqual(harness._inference_ema_sec, 0.020)


class IdleMotionScaleTests(unittest.TestCase):
    """The idle loop's own motion decides how restless the avatar looks."""

    class _Harness:
        advance = BaseAvatar._advance_idle_index

        def __init__(self, scale):
            self.idle_motion_scale = scale

    def _advance(self, scale: float, frames: int) -> int:
        harness = self._Harness(scale)
        index, carry = 0, 0.0
        for _ in range(frames):
            index, carry = harness.advance(index, carry)
        return index

    def test_full_speed_advances_one_source_frame_per_output_frame(self):
        self.assertEqual(self._advance(1.0, 100), 100)

    def test_half_speed_halves_the_motion_frequency(self):
        self.assertEqual(self._advance(0.5, 100), 50)

    def test_uneven_scale_does_not_drift(self):
        # 1/3 speed over 300 frames must land on exactly 100, not 99 or 101.
        self.assertEqual(self._advance(1 / 3, 300), 100)

    def test_index_never_moves_backwards_or_skips(self):
        harness = self._Harness(0.4)
        index, carry, seen = 0, 0.0, []
        for _ in range(50):
            index, carry = harness.advance(index, carry)
            seen.append(index)
        steps = set(np.diff([0] + seen))
        self.assertTrue(steps <= {0, 1}, steps)


class CodecTuningTests(unittest.TestCase):
    def setUp(self):
        from aiortc.codecs import h264, vpx

        self.h264, self.vpx = h264, vpx
        self._saved = (
            h264.DEFAULT_BITRATE, h264.MAX_BITRATE,
            vpx.DEFAULT_BITRATE, vpx.MAX_BITRATE,
        )

    def tearDown(self):
        (self.h264.DEFAULT_BITRATE, self.h264.MAX_BITRATE,
         self.vpx.DEFAULT_BITRATE, self.vpx.MAX_BITRATE) = self._saved

    def test_new_encoders_start_at_the_configured_default(self):
        apply_webrtc_bitrate(3_000_000, 8_000_000)
        self.assertEqual(self.h264.H264Encoder().target_bitrate, 3_000_000)

    def test_remb_feedback_is_clamped_to_the_configured_ceiling(self):
        apply_webrtc_bitrate(3_000_000, 8_000_000)
        encoder = self.h264.H264Encoder()
        encoder.target_bitrate = 50_000_000
        self.assertEqual(encoder.target_bitrate, 8_000_000)

    def test_vp8_fallback_is_raised_alongside_h264(self):
        apply_webrtc_bitrate(3_000_000, 8_000_000)
        self.assertEqual(self.vpx.MAX_BITRATE, 8_000_000)

    def test_ceiling_above_the_advertised_h264_level_is_clamped(self):
        applied = apply_webrtc_bitrate(3_000_000, 50_000_000)
        self.assertEqual(applied["max_bitrate"], H264_LEVEL_31_MAX_BITRATE)

    def test_default_above_ceiling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "video_bitrate"):
            apply_webrtc_bitrate(9_000_000, 8_000_000)

    def test_non_positive_bitrate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_webrtc_bitrate(0, 8_000_000)


if __name__ == "__main__":
    unittest.main()
