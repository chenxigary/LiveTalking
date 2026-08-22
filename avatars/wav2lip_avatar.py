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
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

# The bundled wav2lip256 checkpoint has a fixed 256x256 visual input.  Older
# avatar generators defaulted to 96, so normalise legacy assets at load time
# instead of letting the model fail deep inside its encoder.
WAV2LIP_INPUT_SIZE = 256

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)

    input_img_list = sorted(
        glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    # Decode once at startup.  Keeping PNG bytes saved memory but forced a
    # full 890x1920 PNG decode for every 25-fps idle frame, consuming roughly
    # two CPU cores and starving local LLM/TTS on Apple Silicon.  This machine
    # has enough unified memory for the ~0.7 GB decoded frame cycle.
    frame_list_cycle = read_imgs(input_img_list)

    input_face_list = sorted(
        glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    face_list_cycle = read_imgs(input_face_list)
    legacy_sizes = {tuple(frame.shape[:2]) for frame in face_list_cycle}
    if legacy_sizes != {(WAV2LIP_INPUT_SIZE, WAV2LIP_INPUT_SIZE)}:
        logger.warning(
            "Resizing Wav2Lip face assets from %s to %dx%d; regenerate the "
            "avatar at 256 for best quality",
            sorted(legacy_sizes),
            WAV2LIP_INPUT_SIZE,
            WAV2LIP_INPUT_SIZE,
        )
        face_list_cycle = [
            cv2.resize(
                frame,
                (WAV2LIP_INPUT_SIZE, WAV2LIP_INPUT_SIZE),
                interpolation=cv2.INTER_LANCZOS4,
            )
            for frame in face_list_cycle
        ]

    return frame_list_cycle, face_list_cycle, coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        raw_frame = self.frame_list_cycle[idx]
        if isinstance(raw_frame, bytes):
            combine_frame = cv2.imdecode(np.frombuffer(raw_frame, np.uint8), cv2.IMREAD_COLOR)
        else:
            combine_frame = raw_frame.copy()

        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame
