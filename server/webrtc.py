###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
# Fallback only. The real rate is per-track (PlayerStreamTrack.sample_rate)
# because the Avatar's audio output rate is configurable: lip features are
# locked to 16kHz, the listener's audio is not.
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind, sample_rate: int = SAMPLE_RATE):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self.sample_rate = int(sample_rate)
        self.audio_time_base = fractions.Fraction(1, self.sample_rate)
        self._queue = queue.Queue(maxsize=100)
        self._generation = 0
        self._generation_lock = threading.RLock()
        self._continuity_lock = threading.RLock()
        self._audio_continuity_armed = False
        self._audio_speech_active = False
        self._audio_fallback_events = 0
        self._audio_fallback_frames = 0
        self._audio_current_fallback_frames = 0
        self._audio_max_fallback_frames = 0
        self._audio_speech_fallback_events = 0
        self._audio_speech_fallback_frames = 0
        self._audio_current_speech_fallback_frames = 0
        self._audio_max_speech_fallback_frames = 0
        self._sender_timing_lock = threading.RLock()
        self._sender_timing_frames = 0
        self._sender_last_monotonic_ms = None
        self._sender_last_media_ms = None
        self._sender_gap_events = []
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                target = self._start + self.current_frame_count * VIDEO_PTIME
                now = time.time()
                # Queue starvation must not be followed by a catch-up burst.
                # Keep media PTS continuous but move the wall-clock pacing
                # anchor forward after a missed output interval.
                if now - target > VIDEO_PTIME:
                    self._start += now - target
                    target = now
                wait = target - now
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * self.sample_rate)
                self.current_frame_count += 1
                target = self._start + self.current_frame_count * AUDIO_PTIME
                now = time.time()
                if now - target > AUDIO_PTIME:
                    self._start += now - target
                    target = now
                wait = target - now
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, self.audio_time_base

    async def recv(self) -> Union[Frame, Packet]:
        recv_entry_monotonic_ms = time.monotonic_ns() / 1_000_000
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        used_audio_fallback = False
        while True:
            try:
                item = self._queue.get_nowait()
                if len(item) == 3:
                    frame, eventpoint, generation = item
                else:
                    frame, eventpoint = item
                    generation = self.generation
                try:
                    self._queue.task_done()
                except ValueError:
                    pass
                if generation < self.generation:
                    continue
                break
            except queue.Empty:
                if self.kind == "audio" and self._audio_continuity_armed:
                    samples = round(AUDIO_PTIME * self.sample_rate)
                    audio = np.zeros((1, samples), dtype=np.int16)
                    frame = AudioFrame.from_ndarray(
                        audio,
                        layout="mono",
                        format="s16",
                    )
                    frame.sample_rate = self.sample_rate
                    eventpoint = None
                    generation = self.generation
                    used_audio_fallback = True
                    self._record_audio_fallback()
                    break
                await asyncio.sleep(0.005)

        queue_ready_monotonic_ms = time.monotonic_ns() / 1_000_000

        if self.kind == "audio" and not used_audio_fallback:
            status = str(eventpoint.get("status", "")) if isinstance(eventpoint, dict) else ""
            if status in {"start", "start_end"}:
                with self._continuity_lock:
                    self._audio_speech_active = True
            self._record_real_audio_frame()
            if status in {"end", "start_end"}:
                with self._continuity_lock:
                    self._audio_speech_active = False
                
        pts, time_base = await self.next_timestamp()
        pacing_done_monotonic_ms = time.monotonic_ns() / 1_000_000
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        self._record_sender_timing(
            pts,
            time_base,
            recv_entry_monotonic_ms=recv_entry_monotonic_ms,
            queue_ready_monotonic_ms=queue_ready_monotonic_ms,
            pacing_done_monotonic_ms=pacing_done_monotonic_ms,
        )
        return frame

    def _record_sender_timing(
        self,
        pts,
        time_base,
        *,
        now_monotonic_ms=None,
        recv_entry_monotonic_ms=None,
        queue_ready_monotonic_ms=None,
        pacing_done_monotonic_ms=None,
    ) -> None:
        """Record when a frame is handed back to aiortc's RTP sender.

        ``time.monotonic_ns`` is system-wide on macOS, so a separate local
        receiver can compare these intervals without wall-clock sync.  Keep
        only the eight largest events to make the per-session snapshot bounded.
        """

        now_ms = (
            time.monotonic_ns() / 1_000_000
            if now_monotonic_ms is None
            else float(now_monotonic_ms)
        )
        media_ms = float(pts * time_base * 1000)
        recv_entry_ms = (
            now_ms
            if recv_entry_monotonic_ms is None
            else float(recv_entry_monotonic_ms)
        )
        queue_ready_ms = (
            recv_entry_ms
            if queue_ready_monotonic_ms is None
            else float(queue_ready_monotonic_ms)
        )
        pacing_done_ms = (
            queue_ready_ms
            if pacing_done_monotonic_ms is None
            else float(pacing_done_monotonic_ms)
        )
        with self._sender_timing_lock:
            self._sender_timing_frames += 1
            if (
                self._sender_last_monotonic_ms is not None
                and self._sender_last_media_ms is not None
            ):
                wall_gap_ms = now_ms - self._sender_last_monotonic_ms
                media_gap_ms = media_ms - self._sender_last_media_ms
                event = {
                    "gap_ms": wall_gap_ms,
                    "media_gap_ms": media_gap_ms,
                    "excess_gap_ms": wall_gap_ms - media_gap_ms,
                    "start_monotonic_ms": self._sender_last_monotonic_ms,
                    "end_monotonic_ms": now_ms,
                    "inter_call_gap_ms": (
                        recv_entry_ms - self._sender_last_monotonic_ms
                    ),
                    "queue_wait_ms": queue_ready_ms - recv_entry_ms,
                    "pacing_wait_ms": pacing_done_ms - queue_ready_ms,
                    "post_pacing_ms": now_ms - pacing_done_ms,
                    "recv_total_ms": now_ms - recv_entry_ms,
                }
                self._sender_gap_events.append(event)
                self._sender_gap_events.sort(
                    key=lambda item: item["gap_ms"], reverse=True
                )
                del self._sender_gap_events[8:]
            self._sender_last_monotonic_ms = now_ms
            self._sender_last_media_ms = media_ms

    def sender_timing_snapshot(self) -> dict:
        expected_interval_ms = round(
            (AUDIO_PTIME if self.kind == "audio" else VIDEO_PTIME) * 1000
        )
        with self._sender_timing_lock:
            events = [dict(item) for item in self._sender_gap_events]
            return {
                "clock": "time.monotonic_ns",
                "kind": self.kind,
                "expected_interval_ms": expected_interval_ms,
                "frames_returned": self._sender_timing_frames,
                "gap_max_ms": None if not events else events[0]["gap_ms"],
                "gap_events": events,
            }

    def _record_real_audio_frame(self) -> None:
        with self._continuity_lock:
            recovered_frames = self._audio_current_fallback_frames
            recovered_speech_frames = self._audio_current_speech_fallback_frames
            self._audio_continuity_armed = True
            self._audio_current_fallback_frames = 0
            self._audio_current_speech_fallback_frames = 0
        if recovered_frames:
            mylogger.info(
                "WebRTC audio queue recovered after %dms fallback silence (%dms during speech)",
                recovered_frames * round(AUDIO_PTIME * 1000),
                recovered_speech_frames * round(AUDIO_PTIME * 1000),
            )

    def _record_audio_fallback(self) -> None:
        with self._continuity_lock:
            if self._audio_current_fallback_frames == 0:
                self._audio_fallback_events += 1
                mylogger.warning("WebRTC audio queue empty; emitting paced silence")
            self._audio_fallback_frames += 1
            self._audio_current_fallback_frames += 1
            self._audio_max_fallback_frames = max(
                self._audio_max_fallback_frames,
                self._audio_current_fallback_frames,
            )
            if self._audio_speech_active:
                if self._audio_current_speech_fallback_frames == 0:
                    self._audio_speech_fallback_events += 1
                self._audio_speech_fallback_frames += 1
                self._audio_current_speech_fallback_frames += 1
                self._audio_max_speech_fallback_frames = max(
                    self._audio_max_speech_fallback_frames,
                    self._audio_current_speech_fallback_frames,
                )

    def continuity_snapshot(self) -> dict:
        frame_ms = round(AUDIO_PTIME * 1000)
        with self._continuity_lock:
            return {
                "armed": self._audio_continuity_armed,
                "fallback_events": self._audio_fallback_events,
                "fallback_silence_ms": self._audio_fallback_frames * frame_ms,
                "current_fallback_silence_ms": (
                    self._audio_current_fallback_frames * frame_ms
                ),
                "max_fallback_silence_ms": self._audio_max_fallback_frames * frame_ms,
                "speech_fallback_events": self._audio_speech_fallback_events,
                "speech_fallback_silence_ms": (
                    self._audio_speech_fallback_frames * frame_ms
                ),
                "current_speech_fallback_silence_ms": (
                    self._audio_current_speech_fallback_frames * frame_ms
                ),
                "max_speech_fallback_silence_ms": (
                    self._audio_max_speech_fallback_frames * frame_ms
                ),
            }
    
    def stop(self):
        super().stop()
        self.clear_queue()
        if self._player is not None:
            self._player._stop(self)
            self._player = None

    @property
    def generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def set_generation(self, generation: int) -> None:
        with self._generation_lock:
            self._generation = max(self._generation, int(generation))

    def clear_queue(self) -> int:
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return drained
            else:
                drained += 1
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

def player_worker_thread(
    quit_event,
    container
):
    container.render(quit_event)

class HumanPlayer:

    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.audio_sample_rate = int(
            getattr(avatar_session, "output_sample_rate", SAMPLE_RATE)
        )
        self.__audio = PlayerStreamTrack(
            self, kind="audio", sample_rate=self.audio_sample_rate
        )
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = avatar_session
        self.__generation_lock = threading.RLock()
        self.__loop_watchdog_lock = threading.RLock()
        self.__loop_watchdog_handle = None
        self.__loop_watchdog_loop = None
        self.__loop_watchdog_interval_s = 0.020
        self.__loop_watchdog_ticks = 0
        self.__loop_watchdog_events = []
        self.__generation = int(getattr(avatar_session, "generation", 0))
        self.__audio.set_generation(self.__generation)
        self.__video.set_generation(self.__generation)
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self

    def push_video(self, frame):
        self.push_video_for_generation(frame, self.generation)

    def push_video_for_generation(self, frame, generation: int):
        if int(generation) != self.generation:
            return
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        self._put_bounded(self.__video._queue, (new_frame, None, int(generation)))

    def push_audio(self, frame, eventpoint=None):
        from av import AudioFrame
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = self.audio_sample_rate
        metadata_generation = self.generation
        if isinstance(eventpoint, dict) and eventpoint.get("generation") is not None:
            try:
                metadata_generation = int(eventpoint["generation"])
            except (TypeError, ValueError):
                metadata_generation = self.generation
        if metadata_generation < self.generation:
            return
        self._put_bounded(
            self.__audio._queue,
            (new_frame, eventpoint, metadata_generation),
        )

    @property
    def generation(self) -> int:
        with self.__generation_lock:
            return self.__generation

    def set_generation(self, generation: int) -> None:
        with self.__generation_lock:
            self.__generation = max(self.__generation, int(generation))
            current = self.__generation
        self.__audio.set_generation(current)
        self.__video.set_generation(current)

    def clear_queues(self) -> None:
        self.__audio.clear_queue()
        self.__video.clear_queue()

    def continuity_snapshot(self) -> dict:
        snapshot = self.__audio.continuity_snapshot()
        snapshot["sender_timing"] = {
            "audio": self.__audio.sender_timing_snapshot(),
            "video": self.__video.sender_timing_snapshot(),
            "event_loop": self._event_loop_timing_snapshot(),
        }
        return snapshot

    def _ensure_loop_watchdog(self) -> None:
        """Sample event-loop scheduling lag without changing media pacing.

        A long interval between two calls to ``track.recv`` is ambiguous: it
        can be time spent in aiortc's executor-backed encoder, transport send,
        or a starved asyncio loop.  This independent 20-ms heartbeat makes
        those cases distinguishable while retaining only bounded telemetry.
        """

        with self.__loop_watchdog_lock:
            if self.__loop_watchdog_handle is not None:
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self.__loop_watchdog_loop = loop
            expected_ns = time.monotonic_ns() + round(
                self.__loop_watchdog_interval_s * 1_000_000_000
            )
            self.__loop_watchdog_handle = loop.call_later(
                self.__loop_watchdog_interval_s,
                self._loop_watchdog_tick,
                expected_ns,
            )

    def _loop_watchdog_tick(self, expected_ns: int) -> None:
        now_ns = time.monotonic_ns()
        lag_ms = max(0.0, (now_ns - expected_ns) / 1_000_000)
        event = {
            "lag_ms": lag_ms,
            "expected_monotonic_ms": expected_ns / 1_000_000,
            "actual_monotonic_ms": now_ns / 1_000_000,
        }
        with self.__loop_watchdog_lock:
            self.__loop_watchdog_ticks += 1
            self.__loop_watchdog_events.append(event)
            self.__loop_watchdog_events.sort(
                key=lambda item: item["lag_ms"], reverse=True
            )
            del self.__loop_watchdog_events[8:]
            loop = self.__loop_watchdog_loop
            if loop is None or loop.is_closed():
                self.__loop_watchdog_handle = None
                return
            next_expected_ns = now_ns + round(
                self.__loop_watchdog_interval_s * 1_000_000_000
            )
            self.__loop_watchdog_handle = loop.call_later(
                self.__loop_watchdog_interval_s,
                self._loop_watchdog_tick,
                next_expected_ns,
            )

    def _event_loop_timing_snapshot(self) -> dict:
        with self.__loop_watchdog_lock:
            events = [dict(item) for item in self.__loop_watchdog_events]
            return {
                "clock": "time.monotonic_ns",
                "interval_ms": round(self.__loop_watchdog_interval_s * 1000),
                "ticks": self.__loop_watchdog_ticks,
                "lag_max_ms": None if not events else events[0]["lag_ms"],
                "lag_events": events,
            }

    def _stop_loop_watchdog(self) -> None:
        with self.__loop_watchdog_lock:
            handle = self.__loop_watchdog_handle
            self.__loop_watchdog_handle = None
            self.__loop_watchdog_loop = None
        if handle is not None:
            handle.cancel()

    @staticmethod
    def _put_bounded(target: queue.Queue, item) -> None:
        """Never let a slow WebRTC consumer block the render thread."""

        try:
            target.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                target.task_done()
            except ValueError:
                pass
        try:
            target.put_nowait(item)
        except queue.Full:
            # A concurrent producer won the single available slot; dropping
            # this frame is preferable to stalling the avatar pipeline.
            pass

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

    def notify(self,eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self._ensure_loop_watchdog()
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
                # A broken third-party inference backend must never keep the
                # whole local server alive after its WebRTC session is gone.
                daemon=True,
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            worker = self.__thread
            self.__thread = None
            # aiortc invokes track.stop() on the aiohttp event-loop thread.
            # Joining the renderer here freezes every HTTP/static route while
            # Wav2Lip winds down (or forever if a backend is stuck).  Reap it
            # off-loop with a bounded daemon waiter instead.
            threading.Thread(
                name="media-player-reaper",
                target=self._reap_worker,
                args=(worker,),
                daemon=True,
            ).start()

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None
        if not self.__started:
            self._stop_loop_watchdog()

    @staticmethod
    def _reap_worker(worker: threading.Thread) -> None:
        worker.join(timeout=5.0)
        if worker.is_alive():
            mylogger.warning(
                "HumanPlayer worker did not stop within 5s; detached daemon thread"
            )

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
