#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, queue, numpy as np
from pathlib import Path
import sounddevice as sd
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter


# ==== 루트 경로 ====
ROOT = Path(__file__).resolve().parents[1]  # new_yam/


# ==== 경로 ====
YAMNET_TFLITE = ROOT / "models/yamnet/yamnet.tflite"
HEAD_TFLITE   = ROOT / "models/head/head_1024_fp16.tflite"
LABELS_TXT    = ROOT / "scripts/data/labels17.txt"

# ==== 오디오 ====
SR  = 16000       # 16kHz
WIN = 16000       # 1.0초 창
HOP = 8000        # 0.5초 stride
VAD_DB = -45      # 너무 조용하면 스킵

# ==== 라벨 ====
with open(LABELS_TXT, "r") as f:
    LABELS = [l.strip() for l in f if l.strip()]

# ==== YAMNet ====
yam = Interpreter(model_path=str(YAMNET_TFLITE), num_threads=2)
yam.allocate_tensors()
in_det = yam.get_input_details()[0]
yam_in_idx = in_det["index"]
# 입력 shape 유연 처리
if (len(in_det['shape']) == 1 and in_det['shape'][0] not in (0, 16000)) \
   or (len(in_det['shape']) == 2 and tuple(in_det['shape']) not in ((1,16000),(16000,1))):
    try:
        yam.resize_tensor_input(in_det['index'], [16000], strict=False)
        yam.allocate_tensors()
    except Exception:
        pass
yam_outs = yam.get_output_details()
YAM_EMB_IDX = 1  # 0: 521 logits, 1: 1024 emb, 2: 64

# ==== Head ====
head = Interpreter(model_path=str(HEAD_TFLITE), num_threads=2)
head.allocate_tensors()
head_in_det  = head.get_input_details()[0]
head_in_idx  = head_in_det["index"]
head_out_idx = head.get_output_details()[0]["index"]

# ==== 유틸 ====
def rms_db(x: np.ndarray) -> float:
    r = np.sqrt(np.mean(x.astype(np.float32)**2) + 1e-12)
    return 20 * np.log10(r + 1e-12)

def entropy(p: np.ndarray) -> float:
    # 멀티라벨 시그모이드의 불확실성 가늠용: Bernoulli 엔트로피 평균
    eps = 1e-12
    h = -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))
    return float(np.mean(h))

def adapt_embed_for_head(emb: np.ndarray) -> None:
    # emb: (T,1024) 또는 (1,1024) 또는 (1024,)
    if emb.ndim == 2:
        emb_vec = emb.mean(axis=0)
    elif emb.ndim == 1 and emb.shape[0] == 1024:
        emb_vec = emb
    else:
        raise ValueError(f"Unexpected YAMNet embedding shape: {emb.shape}")

    x = emb_vec.astype(np.float32, copy=False)
    head_shape = head_in_det['shape']

    if len(head_shape) == 1:
        # (1024,)
        if head_shape[0] > 0 and x.shape[0] != head_shape[0]:
            if x.shape[0] < head_shape[0]:
                x = np.pad(x, (0, head_shape[0]-x.shape[0]))
            else:
                x = x[:head_shape[0]]
        head.set_tensor(head_in_idx, x)

    elif len(head_shape) == 2:
        r, c = head_shape
        if r == 1 and c == x.shape[0]:
            head.set_tensor(head_in_idx, x[np.newaxis, :])
        elif c == 1 and r == x.shape[0]:
            head.set_tensor(head_in_idx, x[:, np.newaxis])
        else:
            head.resize_tensor_input(head_in_idx, [1, x.shape[0]], strict=False)
            head.allocate_tensors()
            head.set_tensor(head_in_idx, x[np.newaxis, :])
    else:
        raise ValueError(f"Unsupported head input rank: {len(head_shape)}")

def infer_one_second(wave: np.ndarray) -> np.ndarray:
    yam.set_tensor(yam_in_idx, wave.astype(np.float32))
    yam.invoke()
    emb = yam.get_tensor(yam_outs[YAM_EMB_IDX]["index"])
    adapt_embed_for_head(emb)
    head.invoke()
    prob = head.get_tensor(head_out_idx)[0]  # (17,)
    return prob

def fmt_table(prob: np.ndarray, k=None):
    # k=None이면 17개 전부 출력
    idx = np.argsort(-prob)
    if k is not None:
        idx = idx[:k]
    rows = []
    for i in idx:
        rows.append(f"{LABELS[i]:<15} {prob[i]:.3f}")
    return "\n".join(rows)

# ==== Unknown(else) 후처리 ====
TH = 0.50     # per-class threshold
ENT_TH = 0.60 # 평균 Bernoulli 엔트로피 상한

def decide_unknown(prob: np.ndarray):
    active_idx = [i for i, p in enumerate(prob) if p > TH]
    ent = entropy(prob)
    is_unknown = (len(active_idx) == 0) or (ent > ENT_TH)
    # 표시에 쓰기 위한 점수(플래그처럼 0/1)
    unk_score = 1.0 if is_unknown else 0.0
    return is_unknown, unk_score

# ==== 스트리밍 ====
q = queue.Queue()

def audio_cb(indata, frames, time_info, status):
    if status:
        print(status)
    x = indata
    if x.ndim == 2:
        x = x.mean(axis=1, keepdims=True)
    q.put(x[:,0].copy())

def main():
    buf = np.zeros(0, dtype=np.float32)
    print("🎙️ 실시간 추론 시작 (Ctrl+C 종료). 권한 안 뜨면 시스템 설정에서 터미널/파이썬 마이크 허용.")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=HOP, callback=audio_cb):
        while True:
            block = q.get()
            if buf.size == 0:
                buf = block
            else:
                buf = np.concatenate([buf, block], axis=0)
                if buf.size > WIN:
                    buf = buf[-WIN:]

            if buf.size < WIN:
                continue

            if rms_db(buf) < VAD_DB:
                continue

            prob = infer_one_second(buf)
            ts = time.strftime("%H:%M:%S")

            # --- 출력: 17개 전부 + unknown 1줄 = 18줄 ---
            print(f"\n[{ts}] 17-class probabilities + unknown")
            print(fmt_table(prob, k=None))  # 전부 출력
            is_unk, unk_score = decide_unknown(prob)
            print(f"{'unknown':<15} {unk_score:.3f}")

if __name__ == "__main__":
    for p in [YAMNET_TFLITE, HEAD_TFLITE, LABELS_TXT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"경로 확인: {p}")
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료.")
