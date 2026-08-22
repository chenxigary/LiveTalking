#
#  HuBERT 音频特征提取 — 用于 Wav2LipLS / UltraLight
#  迁移自 hubertasr.py
#

import time
import torch
import numpy as np
from avatars.audio_features.base_asr import BaseASR
from avatars.ultralight.audio2feature import Audio2Feature

# hubert audio feature
class HubertASR(BaseASR):
    #audio_feat_length: select audio feature before and after
    def __init__(self, opt, parent, audio_processor:Audio2Feature, audio_feat_length=[8,8]):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
        #self.stride_left_size = 32
        #self.stride_right_size = 32
        self.audio_feat_length = audio_feat_length
        self.last_is_silence = True


    def run_step(self):
        start_time = time.time()

        generation, inputs, audio_frames = self.collect_step_audio(self.batch_size * 2)
        if inputs is None:
            return

        is_all_silence = all(frame.type != 0 for frame in audio_frames)
        mel_chunks = self.batch_size*[np.zeros((10,1024),dtype=np.float32)]  # default empty feature for silence
        if not is_all_silence or not self.last_is_silence: 
            inputs = np.concatenate(self.frames)  # [N * chunk]

            mel = self.audio_processor.get_hubert_from_16k_speech(inputs)
            mel_chunks = self._feature2chunks(feature_array=mel,batch_size=self.batch_size,
                                            audio_feat_win = self.audio_feat_length, start=self.stride_left_size/2,
                                            feature_idx_multiplier=2)

        if self.publish_step_feature(mel_chunks, generation):
            with self._state_lock:
                if generation == self._generation:
                    self.last_is_silence = is_all_silence
        #print(f"Processing audio costs {(time.time() - start_time) * 1000}ms")
        #return is_all_silence and self.last_is_silence
