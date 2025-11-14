#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


#경로 수정 필요
CSV_PATH = "data/embeddings_1024/multilabel_emb.csv"
OUT_PATH = "runs_multi/best_1024.keras"
NUM_CLASSES = 17
SEED = 42

def parse_vector_field(val):
    """
    emb_npy 필드 자동 파싱:
    - [0.1, 0.2, ...] 같은 리스트 문자열 -> 안전하게 ast.literal_eval
    - '0.1 0.2 ...' 또는 '0.1,0.2,...' 같은 숫자 나열 -> 구분자 정규화 후 fromstring
    - '/path/xxx.npy' -> np.load (2D면 time-mean으로 1D화)
    """
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val, dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr.astype(np.float32)

    s = str(val).strip()

    # 1) .npy 파일 경로
    if s.endswith(".npy") and os.path.exists(s):
        arr = np.load(s)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr

    # 2) JSON/리스트 문자열
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = np.array(ast.literal_eval(s), dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            return arr
        except Exception:
            pass  # 아래 일반 파싱으로 폴백

    # 3) 일반 숫자 나열 문자열: 쉼표/줄바꿈 -> 공백 치환 후 fromstring
    s_norm = s.replace(",", " ").replace("\n", " ").replace("\t", " ")
    arr = np.fromstring(s_norm, sep=" ", dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"임베딩 파싱 실패: {s[:120]}...")
    return arr

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if "emb_npy" not in df.columns:
        raise ValueError("CSV에 'emb_npy' 컬럼이 없습니다. cache_embeddings.py 출력 형식을 확인하세요.")

    # 피처/라벨 분리
    label_cols = [c for c in df.columns if c != "emb_npy"]
    y = df[label_cols].values.astype(np.float32)

    # 임베딩 파싱
    X_list = []
    bad = 0
    for i, v in enumerate(df["emb_npy"].values):
        try:
            vec = parse_vector_field(v)
            X_list.append(vec)
        except Exception as e:
            bad += 1
            # 필요하면 문제 행 로깅
            # print(f"[WARN] row {i} parse fail: {e}")

    if bad:
        print(f"[WARN] 파싱 실패 {bad}개 행 제외")

    X = np.stack(X_list)
    y = y[: len(X_list)]

    # 차원 점검(보통 1024)
    emb_dim = X.shape[1]
    print(f"[INFO] 임베딩 차원: {emb_dim}, 샘플 수: {X.shape[0]}")
    return X, y, emb_dim

def build_head(input_dim, num_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='sigmoid'),
    ])

def main():
    X, y, emb_dim = load_dataset(CSV_PATH)

    if emb_dim not in (1024, 2048):
        print(f"[WARN] 예상과 다른 임베딩 차원: {emb_dim}. 그래도 그대로 학습 진행.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=SEED, shuffle=True
    )

    model = build_head(emb_dim, NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        verbose=1
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    model.save(OUT_PATH)
    print(f"[DONE] 저장: {OUT_PATH}")

if __name__ == "__main__":
    main()
