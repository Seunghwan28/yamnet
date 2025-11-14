# inspect_yamnet_tflite.py
import os
import numpy as np
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as tflite

ROOT = Path(__file__).resolve().parents[1]
YAMNET_TFLITE = ROOT / "models/yamnet/yamnet.tflite"

inter = tflite(str(YAMNET_TFLITE))
inter.allocate_tensors()

print("== INPUT ==")
for d in inter.get_input_details():
    print(d["name"], d["shape"], d["dtype"])

print("\n== OUTPUTS ==")
outs = inter.get_output_details()
for i, d in enumerate(outs):
    print(i, d["name"], d["shape"], d["dtype"])

# 어느 출력이 1024 임베딩인지 자동 탐색
emb_idx = None
for i, d in enumerate(outs):
    shp = d["shape"]
    if len(shp) >= 2 and (shp[-1] == 1024):
        emb_idx = i
        break
print("\nEmbedding output index:", emb_idx)
