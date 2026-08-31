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
#
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event, RLock
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import (
    blend_bgr,
    decode_bgr,
    draw_debug_label_bgr,
    mirror_index,
    read_imgs,
)

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

# Share of the 40ms frame budget an inference may consume before the adaptive
# controller widens the stride.  The margin covers paste-back, H264 encode and
# scheduler jitter, which all land on the same deadline.
_STRIDE_SAFETY = 1.25
# EMA weight for measured inference latency: about five frames (200ms at
# 25fps) to react, which tracks a TTS burst without flapping every frame.
_STRIDE_EMA_ALPHA = 0.2


@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)
    generation: int = 0
    # Same 20ms span as ``data`` but at the Avatar's output rate. ``data`` is
    # pinned to 16kHz because that is what the lip model's mel front-end
    # requires; there is no reason for the listener to inherit that ceiling.
    # ``None`` means a legacy producer supplied only the feature-rate frame.
    output: NDArray[np.float32] | None = None

class BaseAvatar:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms)
        # Qwen3-TTS emits 24kHz and 1.19% of its measured power sits in the
        # 8-12kHz band a 16kHz stream discards -- the sibilance that separates
        # "podcast" from "phone call". Features stay at 16kHz regardless.
        self.output_sample_rate = int(getattr(opt, "audio_output_rate", 24000))
        if self.output_sample_rate % (opt.fps * 2):
            raise ValueError(
                f"audio_output_rate {self.output_sample_rate} must divide into "
                f"20ms frames at fps={opt.fps}"
            )
        self.output_chunk = self.output_sample_rate // (opt.fps * 2)
        self.sessionid = self.opt.sessionid

        # The HTTP route and render loop run on different threads.  This lock
        # serializes generation changes with ASR frame ingestion; model
        # inference itself is intentionally not cancelled and is filtered by
        # generation when it returns.
        self._pipeline_lock = RLock()
        self._generation = 0
        # /humanaudio uploads may end at a non-20ms boundary (for example a
        # 24kHz 50ms source chunk becomes 800 samples at 16kHz).  Keep that
        # remainder across streaming uploads instead of dropping it on every
        # request.  Generation/turn changes clear it synchronously.
        # The tail is kept at the *source* rate so that the 16kHz feature
        # stream and the output stream are both derived from one 20ms-aligned
        # buffer. Resampling each independently would let their frame counts
        # drift apart across uploads.
        self._audio_tail = np.empty(0, dtype=np.float32)
        self._audio_tail_rate = 0
        self._audio_tail_generation = 0
        self._audio_tail_turn_id = None
        self._audio_tail_pts_ms = 0
        self._audio_start_pending = False
        self._audio_end_flushed = False
        self._audio_stream_initialized = False

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.msgqueues = []
        # self.custom_opt = {}
        self.__loadcustom()

        self.batch_size = opt.batch_size
        requested_stride = max(1, int(getattr(opt, "inference_stride", 1)))
        # The stride implementation deliberately targets the low-latency
        # batch=1 Apple-Silicon path. A batched model already amortises model
        # overhead and needs a different temporal sampling strategy.
        if requested_stride > 1 and self.batch_size != 1:
            logger.warning(
                "inference_stride=%d requires batch_size=1; disabling stride",
                requested_stride,
            )
            requested_stride = 1
        self.inference_stride = requested_stride
        mode = str(getattr(opt, "inference_stride_mode", "adaptive")).strip().lower()
        if mode not in {"adaptive", "fixed"}:
            raise ValueError("inference_stride_mode must be 'adaptive' or 'fixed'")
        # Stride trades lip-sync fidelity for throughput: one inferred pose is
        # held across `stride` frames, so 3/4 of the frames carry no mouth
        # change. It exists because inference also gates audio delivery — this
        # loop emits `stride` frames' worth of audio per model call — so a
        # stride too small for the hardware breaks the voice, not just the lips.
        # Adaptive mode therefore runs at 1 and only degrades (up to
        # `inference_stride`) when measured latency says it must.
        self.inference_stride_mode = mode
        self._active_stride = 1 if mode == "adaptive" else requested_stride
        self._inference_ema_sec = 0.0
        # Monotonic render telemetry. Read without a lock on purpose: these
        # only ever increase, and a status read that races a render step is
        # off by one frame, which is meaningless at 25fps.
        self._inference_samples = 0
        self._speaking_frames = 0
        self._audio_push_logged = False
        logger.info(
            "inference stride: mode=%s limit=%d start=%d",
            self.inference_stride_mode,
            self.inference_stride,
            self._active_stride,
        )
        self.debug_label = bool(getattr(opt, "debug_label", False))
        # Idle playback speed. The source loop's own motion sets how restless
        # the avatar looks between turns; this project's clip measures 2.76 Hz
        # of head movement, which reads as fidgeting rather than breathing.
        # Holding each source frame longer during silence lowers that
        # frequency without new material. Speaking frames are unaffected.
        self.idle_motion_scale = float(getattr(opt, "idle_motion_scale", 1.0))
        if not 0.05 <= self.idle_motion_scale <= 1.0:
            raise ValueError("idle_motion_scale must be between 0.05 and 1.0")
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()

        _tts_modules = {
            'edgetts': 'tts.edge',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts',
            'omnitts': 'tts.omnitts'
        }

        if opt.tts in _tts_modules:
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    @property
    def generation(self) -> int:
        with self._pipeline_lock:
            return self._generation

    @staticmethod
    def _drain_queue(q) -> int:
        """Drain a queue through its synchronized public API."""

        drained = 0
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return drained
            else:
                drained += 1
                try:
                    q.task_done()
                except ValueError:
                    pass

    @staticmethod
    def _coerce_generation(value, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            generation = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation must be an integer") from exc
        if generation < 0:
            raise ValueError("generation must be non-negative")
        return generation

    def _clear_transport_locked(self, generation: int) -> None:
        """Invalidate queued WebRTC media without stopping its worker."""

        output = getattr(self, "output", None)
        player = getattr(output, "_player", None)
        if player is None:
            return
        if hasattr(player, "set_generation"):
            player.set_generation(generation)
        if hasattr(player, "clear_queues"):
            player.clear_queues()

    def _clear_pipeline_locked(self, generation: int) -> None:
        """Clear all in-memory stages and invalidate their old generation."""

        self._generation = generation
        self._clear_audio_tail_locked()

        if hasattr(self, "tts"):
            # BaseTTS historically used queue.queue.clear(), which is not
            # thread-safe. Drain through Queue.get and transition the provider
            # to PAUSE without calling that legacy clear implementation.
            msgqueue = getattr(self.tts, "msgqueue", None)
            if msgqueue is not None:
                self._drain_queue(msgqueue)
            state = getattr(self.tts, "state", None)
            pause = getattr(type(state), "PAUSE", None)
            if pause is not None:
                self.tts.state = pause

        if hasattr(self, "asr"):
            self.asr.set_generation(generation, clear=True)

        self._drain_queue(self.res_frame_queue)
        self.custom_audiotype = 0
        self.speaking = False
        self._clear_transport_locked(generation)

    def _clear_audio_tail_locked(self) -> None:
        """Drop an incomplete streaming WAV tail during a pipeline reset."""

        self._audio_tail = np.empty(0, dtype=np.float32)
        self._audio_tail_rate = 0
        self._audio_tail_generation = self._generation
        self._audio_tail_turn_id = None
        self._audio_tail_pts_ms = 0
        self._audio_start_pending = False
        self._audio_end_flushed = False
        self._audio_stream_initialized = False

    def _accept_generation_locked(self, generation: int) -> bool:
        if generation < self._generation:
            return False
        if generation > self._generation:
            self._clear_pipeline_locked(generation)
        return True

    def _generation_is_current(self, generation: int) -> bool:
        with self._pipeline_lock:
            return generation == self._generation

    def _put_result_frame(self, item, generation: int) -> bool:
        """Bounded, generation-aware enqueue for the renderer queue."""

        while self._generation_is_current(generation):
            try:
                self.res_frame_queue.put(item, timeout=0.05)
                return self._generation_is_current(generation)
            except queue.Full:
                continue
        return False

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict | None = None):
        with self._pipeline_lock:
            metadata = dict(datainfo or {})
            metadata.setdefault("generation", self._generation)
            if hasattr(self, 'tts'):
                self.tts.put_msg_txt(msg, metadata)
            return True
    
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict | None = None, output_chunk=None): # 16khz 20ms pcm
        with self._pipeline_lock:
            metadata = dict(datainfo or {})
            generation = self._coerce_generation(
                metadata.get("generation"), self._generation
            )
            if not self._accept_generation_locked(generation):
                return False
            metadata["generation"] = generation
            if hasattr(self, 'asr'):
                return self.asr.put_audio_frame(audio_chunk, metadata, output_chunk)
            return False

    def put_audio_file(self, filebyte, datainfo:dict | None = None):
        metadata = dict(datainfo or {})
        with self._pipeline_lock:
            generation = self._coerce_generation(
                metadata.get("generation"), self._generation
            )
            if not self._accept_generation_locked(generation):
                return False

        input_stream = BytesIO(filebyte)
        stream, source_rate = self.__create_bytes_stream(input_stream)
        streaming = bool(metadata.get("streaming"))
        stream = np.asarray(stream, dtype=np.float32).reshape(-1)

        with self._pipeline_lock:
            if generation != self._generation:
                return False

            if not streaming:
                # Legacy/non-streaming clients still treat each upload as a
                # complete utterance.  The v3 adapter always sends streaming
                # metadata, so only that path carries a tail across requests.
                tail = np.empty(0, dtype=np.float32)
                tail_pts_ms = self._metadata_pts_ms(metadata)
                start_pending = True
                end_requested = True
                turn_id = metadata.get("turn_id")
            else:
                turn_id = metadata.get("turn_id")
                normalized_turn = None if turn_id is None else str(turn_id)
                if (
                    not self._audio_stream_initialized
                    or self._audio_tail_generation != generation
                    or self._audio_tail_turn_id != normalized_turn
                    or self._audio_tail_rate != source_rate
                ):
                    # A turn switch without an explicit end is a malformed or
                    # interrupted stream.  Never combine its tail with the new
                    # turn; the next generation path also reaches here after
                    # clearing it in _clear_pipeline_locked().
                    self._clear_audio_tail_locked()
                    self._audio_stream_initialized = True
                    self._audio_tail_rate = source_rate
                    self._audio_tail_generation = generation
                    self._audio_tail_turn_id = normalized_turn
                    self._audio_start_pending = bool(metadata.get("start"))

                tail = self._audio_tail
                tail_pts_ms = (
                    self._audio_tail_pts_ms
                    if tail.size
                    else self._metadata_pts_ms(metadata)
                )
                start_pending = self._audio_start_pending
                end_requested = bool(metadata.get("end"))

            if tail.size:
                combined = np.concatenate((tail, stream))
            else:
                combined = stream

            if combined.size == 0:
                if streaming and end_requested:
                    self._audio_end_flushed = True
                    self._audio_tail = np.empty(0, dtype=np.float32)
                return True

            first_seq = self._metadata_seq(metadata, "first_seq")
            last_seq = self._metadata_seq(metadata, "last_seq")
            if last_seq < first_seq:
                last_seq = first_seq
            source_chunk = max(1, source_rate // (self.opt.fps * 2))
            full_count = combined.size // source_chunk
            remainder = combined[full_count * source_chunk:]
            if end_requested and remainder.size:
                # End padding happens once, after all cross-upload tails have
                # been appended.  This preserves the last partial audio while
                # keeping LiveTalking's 20ms frame contract.
                combined = np.pad(
                    combined,
                    (0, source_chunk - remainder.size),
                    mode="constant",
                )
                full_count = combined.size // source_chunk
                remainder = np.empty(0, dtype=np.float32)

            # One 20ms-aligned source buffer, two derived streams: 16kHz for
            # the lip model's mel front-end, output rate for the listener.
            usable = combined[: full_count * source_chunk]
            feature_stream = self._resample_exact(
                usable, source_rate, self.sample_rate, full_count * self.chunk
            )
            output_stream = self._resample_exact(
                usable, source_rate, self.output_sample_rate,
                full_count * self.output_chunk,
            )

            for index in range(full_count):
                eventpoint = dict(metadata)
                status = None
                is_last = index == full_count - 1
                if streaming:
                    if start_pending and not self._audio_end_flushed:
                        status = "start_end" if end_requested and is_last else "start"
                        start_pending = False
                    elif end_requested and is_last:
                        status = "end"
                    elif is_last:
                        status = "progress"
                else:
                    if index == 0:
                        status = "start_end" if is_last else "start"
                    elif is_last:
                        status = "end"
                if status is not None:
                    eventpoint["status"] = status
                    eventpoint["first_seq"] = first_seq
                    eventpoint["last_seq"] = last_seq
                    eventpoint["seq"] = (
                        first_seq
                        if status in {"start", "start_end"}
                        else last_seq
                    )
                eventpoint["generation"] = generation
                eventpoint["pts_ms"] = tail_pts_ms + index * self._frame_duration_ms()
                if not self.put_audio_frame(
                    feature_stream[index * self.chunk:(index + 1) * self.chunk],
                    eventpoint,
                    output_stream[
                        index * self.output_chunk:(index + 1) * self.output_chunk
                    ],
                ):
                    if streaming:
                        self._clear_audio_tail_locked()
                    return False

            if streaming:
                self._audio_start_pending = start_pending
                self._audio_tail = remainder.astype(np.float32, copy=True)
                self._audio_tail_rate = source_rate
                self._audio_tail_generation = generation
                self._audio_tail_turn_id = (
                    None if turn_id is None else str(turn_id)
                )
                self._audio_tail_pts_ms = tail_pts_ms + full_count * self._frame_duration_ms()
                if end_requested:
                    self._audio_tail = np.empty(0, dtype=np.float32)
                    self._audio_end_flushed = True
                    self._audio_start_pending = False
                else:
                    self._audio_end_flushed = False
            return True

    def put_audio_filepath(self, filepath, datainfo:dict | None = None):
        metadata = dict(datainfo or {})
        with self._pipeline_lock:
            generation = self._coerce_generation(
                metadata.get("generation"), self._generation
            )
            if not self._accept_generation_locked(generation):
                return False

        stream, source_rate = self.__create_bytes_stream(filepath)
        stream = self._resample_exact(
            stream, source_rate, self.sample_rate,
            round(stream.size * self.sample_rate / source_rate),
        )
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = dict(metadata)
            status = None
            if first:
                status = 'start'
                first = False
            if streamlen - self.chunk < self.chunk:
                status = 'end'
            if status is not None:
                eventpoint['status'] = status
            if eventpoint.get('pts_ms') is not None:
                try:
                    eventpoint['pts_ms'] = int(eventpoint['pts_ms']) + int(
                        idx * 1000 / self.sample_rate
                    )
                except (TypeError, ValueError):
                    eventpoint['pts_ms'] = 0
            eventpoint['generation'] = generation
            if not self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint):
                return False
            streamlen -= self.chunk
            idx += self.chunk
        return True

    @staticmethod
    def _metadata_pts_ms(metadata: dict) -> int:
        value = metadata.get("pts_ms", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _metadata_seq(metadata: dict, key: str) -> int:
        value = metadata.get(key, metadata.get("seq", 0))
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _frame_duration_ms(self) -> int:
        return max(1, round(self.chunk * 1000 / self.sample_rate))
    
    def __create_bytes_stream(self, byte_stream):
        """Decode to mono float32 at the *source* rate.

        Resampling is deliberately left to the caller: ``put_audio_file``
        needs one 20ms-aligned source buffer to derive both the 16kHz feature
        stream and the output stream from, so that their frame counts cannot
        drift apart.
        """

        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        return stream, int(sample_rate)

    @staticmethod
    def _resample_exact(samples, source_rate: int, target_rate: int, expected: int):
        """Resample to an exact sample count.

        resampy rounds its output length, but the 20ms frame contract needs
        exactly ``expected`` samples or the feature and output streams stop
        lining up frame for frame.
        """

        if source_rate == target_rate or samples.size == 0:
            resampled = samples
        else:
            resampled = resampy.resample(
                x=samples, sr_orig=source_rate, sr_new=target_rate
            )
        if resampled.size > expected:
            resampled = resampled[:expected]
        elif resampled.size < expected:
            resampled = np.pad(resampled, (0, expected - resampled.size))
        return np.asarray(resampled, dtype=np.float32)

    def _to_output_rate(self, samples):
        """Lift a feature-rate frame to the output rate.

        Only legacy producers land here: LiveTalking's built-in TTS plugins
        push 16kHz frames straight into ``put_audio_frame``, and the silence
        and custom-audio paths synthesise them. Silence is by far the common
        case and needs no filter at all.
        """

        if self.output_sample_rate == self.sample_rate:
            return samples
        if not samples.any():
            return np.zeros(self.output_chunk, dtype=np.float32)
        return self._resample_exact(
            samples, self.sample_rate, self.output_sample_rate, self.output_chunk
        )

    def flush_talk(self, generation: int | None = None):
        """Invalidate a turn and clear the complete audio/video pipeline."""

        with self._pipeline_lock:
            if generation is None:
                target = self._generation + 1
            else:
                requested = self._coerce_generation(generation, self._generation)
                # An explicit generation is an idempotent protocol token: a
                # retried control request must not advance the Avatar beyond
                # the Gateway and make its next audio look stale.
                target = max(self._generation, requested)
            self._clear_pipeline_locked(target)
            return target

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def add_msgqueue(self, msgqueue):
        self.msgqueues.append(msgqueue)

    def send_msg(self, msg):
        for q in self.msgqueues:
            q.put(msg)

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            logger.info("notify:%s", eventpoint)
            self.send_msg(json.dumps(eventpoint))

    def start_recording(self):
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', str(self.output_sample_rate),
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tobytes()) #tostring()

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tobytes())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        
        record_path = os.path.join('data', 'record')
        os.makedirs(record_path, exist_ok=True)
        output_file = os.path.join(record_path, f"{self.opt.sessionid}.mp4")
        
        temp_aac = f"temp{self.opt.sessionid}.aac"
        temp_mp4 = f"temp{self.opt.sessionid}.mp4"
        
        cmd_combine_audio = f"ffmpeg -y -i {temp_aac} -i {temp_mp4} -c:v copy -c:a copy {output_file}"
        os.system(cmd_combine_audio)
        
        # 删除临时文件
        try:
            os.remove(temp_aac)
            os.remove(temp_mp4)
        except Exception as e:
            logger.error(f"Error removing temp files: {e}")

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1

    @staticmethod
    def _materialize_video_frame(frame):
        """Return an owned BGR ndarray for cached bytes or decoded frames.

        The macOS memory-saving Wav2Lip loader keeps full avatar frames as
        encoded bytes. Speaking frames are decoded by ``paste_back_frame``,
        while silent frames pass through this base renderer directly.
        Normalising both representations here keeps OpenCV and WebRTC from
        receiving raw ``bytes`` objects.
        """
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return decode_bgr(bytes(frame))
        if isinstance(frame, np.ndarray):
            return frame.copy()
        array = np.asarray(frame)
        if array.ndim != 3:
            raise TypeError(f"Unsupported avatar video frame type: {type(frame).__name__}")
        return array.copy()
        
    def render_snapshot(self) -> dict:
        """Report what the renderer actually did, for probes and diagnostics.

        ``held_frames`` is the count the outside world cannot measure: a
        stride wider than 1 emits several video frames per model call, and box
        jitter makes those repeats non-identical at the pixel level. Only the
        renderer knows how many frames reused a pose.
        """

        inferences = self._inference_samples
        speaking = self._speaking_frames
        return {
            "audio_output_rate": self.output_sample_rate,
            "audio_output_chunk": self.output_chunk,
            "stride_mode": self.inference_stride_mode,
            "stride_limit": self.inference_stride,
            "active_stride": self._active_stride,
            "inference_ms_avg": round(self._inference_ema_sec * 1000.0, 2),
            "inferences": inferences,
            "speaking_frames": speaking,
            "held_frames": max(0, speaking - inferences),
            "held_ratio": (
                round(max(0, speaking - inferences) / speaking, 4)
                if speaking else None
            ),
        }

    def _advance_idle_index(self, index: int, carry: float) -> tuple[int, float]:
        """Step the idle loop, carrying the fraction so it cannot drift.

        The carry keeps a scale like 1/3 landing on exactly one source frame
        per three output frames over any run length. It is also the blend
        weight toward the next source frame: simply repeating a frame would
        keep every step's full size and just space the steps out, trading a
        fast small motion for a slower lurching one.
        """

        carry += self.idle_motion_scale
        step = int(carry)
        return index + step, carry - step

    def _observe_inference(self, elapsed: float) -> None:
        """Retune the adaptive stride from measured inference latency.

        The controller is deliberately reactive rather than predictive: the
        only thing that matters is whether the model is currently keeping up
        with the 25-fps deadline, and that is exactly what ``elapsed``
        measures.  GPU contention from MLX TTS/ASR or llama.cpp shows up here
        directly.
        """

        self._inference_samples += 1
        if self._inference_samples == 1:
            # The first inference on this thread pays a one-time Metal
            # pipeline compile — measured at 1122ms against a 13.6ms steady
            # state. Seeding the EMA with it costs ~18 frames of needlessly
            # widened stride at the start of the first utterance, so the
            # warm-up sample is discarded rather than smoothed.
            logger.info("discarding cold-start inference sample (%.1fms)",
                        elapsed * 1000.0)
            return
        if self._inference_ema_sec == 0.0:
            self._inference_ema_sec = elapsed
        else:
            self._inference_ema_sec = (
                _STRIDE_EMA_ALPHA * elapsed
                + (1.0 - _STRIDE_EMA_ALPHA) * self._inference_ema_sec
            )
        # The average is telemetry in both modes; only the decision below is
        # adaptive-only, so a fixed-stride A/B still reports its own latency.
        if self.inference_stride_mode != "adaptive":
            return

        budget = 1.0 / max(1, int(getattr(self.opt, "fps", 25)))
        needed = math.ceil(self._inference_ema_sec * _STRIDE_SAFETY / budget)
        target = max(1, min(self.inference_stride, needed))
        if target != self._active_stride:
            logger.info(
                "adaptive stride %d -> %d (inference %.1fms now, %.1fms avg, "
                "frame budget %.1fms)",
                self._active_stride,
                target,
                elapsed * 1000.0,
                self._inference_ema_sec * 1000.0,
                budget * 1000.0,
            )
            self._active_stride = target

    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        # Fractional carry for a slowed idle loop; speaking always steps by 1.
        idle_carry = 0.0
        count = 0
        counttime = 0
        last_speaking = False
        speaking_group = []

        def flush_speaking_group():
            """Infer one representative pose and emit its complete audio span.

            On hardware slower than 25 Wav2Lip inferences/s, coupling every
            40-ms audio span to one model call starves WebRTC and produces
            audible gaps.  A stride of four asks the model for one pose per
            160 ms, repeats that pose across four video frames, and still
            emits all eight 20-ms audio packets in their original order.
            """

            nonlocal index, count, counttime
            if not speaking_group:
                return

            generation = speaking_group[0][0]
            if not self._generation_is_current(generation):
                speaking_group.clear()
                return

            representative = len(speaking_group) // 2
            audiofeat_batch = speaking_group[representative][1]
            t = time.perf_counter()
            pred = self.inference_batch(index + representative, audiofeat_batch)
            elapsed = time.perf_counter() - t
            self._observe_inference(elapsed)
            counttime += elapsed
            count += len(pred)
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # batch_size is guaranteed to be one whenever stride > 1.
            # Copying is unnecessary: paste_back_frame converts the shared
            # prediction to uint8 without mutating it.
            res_frame = pred[0]
            for item_generation, _, item_audio_frames in speaking_group:
                if not self._put_result_frame(
                    (
                        item_generation,
                        res_frame,
                        item_audio_frames,
                        mirror_index(length, index),
                    ),
                    item_generation,
                ):
                    break
                index += 1
            speaking_group.clear()

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            feature_batch = self.asr.get_feature_batch(block=True, timeout=1)
            if feature_batch is None:
                continue
            generation = feature_batch.generation
            audiofeat_batch = feature_batch.data
            if not self._generation_is_current(generation):
                speaking_group.clear()
                continue
                
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            for _ in range(self.batch_size * 2):
                audioframe = self.asr.get_output_frame_for_generation(
                    generation, timeout=1
                )
                if audioframe is None:
                    audio_frames = []
                    break
                if audioframe.type == 0:
                    is_all_silence = False               
                audio_frames.append(audioframe)

            if len(audio_frames) != self.batch_size * 2:
                continue

             # 检测状态变化
            current_speaking = not is_all_silence

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                # Do not strand the final 1..stride-1 speaking frames when
                # speech ends before a complete group has accumulated.
                flush_speaking_group()
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    blend = (
                        (mirror_index(length, index + 1), idle_carry)
                        if idle_carry > 1e-6
                        else None
                    )
                    if not self._put_result_frame(
                        (generation, None, audio_frames[i*2:i*2+2], idx, blend),
                        generation,
                    ):
                        break
                    index, idle_carry = self._advance_idle_index(index, idle_carry)
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                if self._active_stride > 1:
                    speaking_group.append((generation, audiofeat_batch, audio_frames))
                    if len(speaking_group) >= self._active_stride:
                        flush_speaking_group()
                else:
                    # A stride that narrowed mid-utterance can leave a partial
                    # group buffered.  Emit it before this frame so audio keeps
                    # leaving the loop in arrival order.
                    flush_speaking_group()
                    t = time.perf_counter()
                    pred = self.inference_batch(index, audiofeat_batch)

                    elapsed = time.perf_counter() - t
                    self._observe_inference(elapsed)
                    counttime += elapsed
                    count += self.batch_size
                    if count >= 100:
                        logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                        count = 0
                        counttime = 0
                    for i, res_frame in enumerate(pred):
                        if not self._put_result_frame(
                            (
                                generation,
                                res_frame,
                                audio_frames[i*2:i*2+2],
                                mirror_index(length, index),
                            ),
                            generation,
                        ):
                            break
                        index = index + 1
                    
            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    def process_frames(self,quit_event):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        _last_speaking = False
        _transition_start = time.time()
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                result_item = self.res_frame_queue.get(block=True, timeout=1)
                try:
                    self.res_frame_queue.task_done()
                except ValueError:
                    pass
            except queue.Empty:
                continue

            idle_blend = None
            if len(result_item) == 5:
                generation, res_frame, audio_frames, idx, idle_blend = result_item
            elif len(result_item) == 4:
                generation, res_frame, audio_frames, idx = result_item
            else:
                # Backward compatibility for an external producer using the
                # old three-item tuple.
                res_frame, audio_frames, idx = result_item
                generation = getattr(audio_frames[0], "generation", self.generation)
            if not self._generation_is_current(generation):
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                target_frame = self._materialize_video_frame(target_frame)
                if idle_blend is not None:
                    # Cross-fade toward the next source frame so a slowed idle
                    # loop moves slowly rather than in held steps.
                    next_idx, weight = idle_blend
                    target_frame = blend_bgr(
                        target_frame,
                        1.0 - weight,
                        self._materialize_video_frame(
                            self.frame_list_cycle[next_idx]
                        ),
                        weight,
                    )
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = blend_bgr(
                            _last_speaking_frame, 1 - alpha, target_frame, alpha
                        )
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                self._speaking_frames += 1
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = blend_bgr(
                            _last_silent_frame, 1 - alpha, current_frame, alpha
                        )
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            if not self._generation_is_current(generation):
                continue
            if self.debug_label:
                combine_frame = draw_debug_label_bgr(combine_frame)
            
            # 使用统一输出接口推送视频帧
            player = getattr(self.output, "_player", None)
            if player is not None and hasattr(player, "push_video_for_generation"):
                player.push_video_for_generation(combine_frame, generation)
            else:
                self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                if not self._generation_is_current(generation):
                    break
                #frame,type,eventpoint = audio_frame
                samples = audio_frame.output
                if samples is None:
                    samples = self._to_output_rate(audio_frame.data)
                if not self._audio_push_logged:
                    self._audio_push_logged = True
                    logger.info(
                        "first audio frame out: %d samples @ %dHz (source span "
                        "%s, feature span %d)",
                        samples.size, self.output_sample_rate,
                        "carried" if audio_frame.output is not None else "lifted",
                        audio_frame.data.size,
                    )
                frame = (samples * 32767).astype(np.int16)

                # 使用统一输出接口推送音频帧
                self.output.push_audio_frame(frame, audio_frame.userdata)
                self.record_audio_data(frame)
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        while not quit_event.is_set(): 
            t = time.perf_counter()
            # BaseASR snapshots and tags each step's generation internally.
            # Do not hold the Avatar lock across a bounded feature-queue put:
            # inference also needs that lock after consuming the queue, and
            # doing both creates a shutdown/refresh deadlock.
            self.asr.run_step()

            buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
            if buffer_size >= 5:
                logger.debug('sleep qsize=%d', buffer_size)
                time.sleep(0.04 * buffer_size * 0.8)
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()
