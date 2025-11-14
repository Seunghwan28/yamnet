# scripts/run_yamnet_plus_head_tflite.py
#!/usr/bin/env python3
import numpy as np
import soundfile as sf
from pathlib import Path

import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

SR = 16000
ROOT = Path(__file__).resolve().parents[1]

YAM  = ROOT / "models/yamnet/yamnet.tflite"
HEAD = ROOT / "models/head/head_1024_fp16.tflite"

LABELS = [
    "door","dishes","cutlery","chopping","frying","microwave","blender",
    "water_tap","sink","toilet_flush","telephone","chewing","speech",
    "television","footsteps","vacuum","hair_dryer"
]


def load_wav16k(path, need_len):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"sr {sr}!={SR}")
    x = x.astype(np.float32)
    if len(x) < need_len:
        y = np.zeros(need_len, np.float32)
        y[:len(x)] = x
        x = y
    else:
        x = x[:need_len]
    return x


# ---------------- YAMNet ----------------
yam = Interpreter(model_path=str(YAM))
yam.allocate_tensors()
yin = yam.get_input_details()[0]

# 입력 길이 지정 (보통 15600 프레임 사용)
need_len = 15600

# 입력 shape이 (1, N) 타입이면 [1, need_len] 로, 1D면 [need_len] 로 조정
if len(yin["shape"]) == 2:
    yam.resize_tensor_input(yin["index"], [1, need_len])
else:
    yam.resize_tensor_input(yin["index"], [need_len])
yam.allocate_tensors()

youts = yam.get_output_details()
# 1024-d emb가 몇 번째 output인지 확인해서 사용 (보통 1번)
EMB_IDX = 1


# ---------------- Head ----------------
head = Interpreter(model_path=str(HEAD))
head.allocate_tensors()
hin  = head.get_input_details()[0]
hout = head.get_output_details()[0]


def infer_one(wav_path: str):
    x = load_wav16k(wav_path, need_len)

    # YAMNet 추론
    if len(yin["shape"]) == 2:
        # (1, N)
        yam.set_tensor(yin["index"], x[np.newaxis, :])
    else:
        # (N,)
        yam.set_tensor(yin["index"], x)
    yam.invoke()

    emb = yam.get_tensor(youts[EMB_IDX]["index"])  # (1, 1024) or (T,1024 등)

    # 프레임 차원 평균해서 (1,1024) 로 맞추기
    if emb.ndim == 2 and emb.shape[0] != 1:
        emb_vec = emb.mean(axis=0, keepdims=True)  # (1,1024)
    elif emb.ndim == 2:
        emb_vec = emb
    elif emb.ndim == 1:
        emb_vec = emb[np.newaxis, :]
    else:
        raise ValueError(f"Unexpected emb shape: {emb.shape}")

    # Head 입력 shape에 맞게 세팅
    hshape = hin["shape"]
    if len(hshape) == 2:
        # (1, 1024) 형태 가정
        head.set_tensor(hin["index"], emb_vec.astype(np.float32))
    elif len(hshape) == 1:
        # (1024,)
        head.set_tensor(hin["index"], emb_vec.astype(np.float32)[0])
    else:
        raise ValueError(f"Unsupported head input shape: {hshape}")

    head.invoke()
    probs = head.get_tensor(hout["index"]).flatten()  # (17,)
    return probs


if __name__ == "__main__":
    # 여기 실제 wav 경로 넣어라 (테스트용)
    wav = "/ABS/PATH/TO/test.wav"
    if not Path(wav).exists():
        raise FileNotFoundError(f"wav 경로 바꿔: {wav}")

    print("YAMNET:", YAM)
    print("HEAD  :", HEAD)

    p = infer_one(wav)
    top5 = sorted(zip(LABELS, p), key=lambda kv: kv[1], reverse=True)[:5]
    for k, v in top5:
        print(f"{k:14s} {v:.3f}")
