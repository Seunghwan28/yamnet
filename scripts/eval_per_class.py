# scripts/eval_per_class.py
import argparse, json, numpy as np, pandas as pd, tensorflow as tf, os

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def load_embeddings_from_npy_column(series):
    """각 행의 emb_npy 경로를 로드해서 2D numpy array로 합침"""
    embs = []
    for path in series:
        if not os.path.exists(path):
            raise FileNotFoundError(f"임베딩 파일 없음: {path}")
        emb = np.load(path)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embs.append(emb)
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ml_csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.ml_csv)
    label_names = load_labels(args.labels)

    if "emb_npy" not in df.columns:
        raise ValueError("CSV에 emb_npy 열이 없습니다. cache_embeddings.py가 이 형식으로 생성된 게 맞나요?")

    # (N, 1024) 임베딩 불러오기
    X = load_embeddings_from_npy_column(df["emb_npy"])
    print(f"[LOAD] 임베딩 {X.shape}")

    # 라벨 행렬 (N, 17)
    Y = df[label_names].to_numpy(dtype=np.float32)

    # 모델 불러오기
    model = tf.keras.models.load_model(args.model, compile=False)

    # 예측
    P = model.predict(X, verbose=0)

    # per-class AUC 계산
    res = {}
    for i, name in enumerate(label_names):
        y_true, y_pred = Y[:, i], P[:, i]
        if np.all(y_true == 0) or np.all(y_true == 1):
            auc, aupr = None, None
        else:
            auc_m = tf.keras.metrics.AUC(curve="ROC")
            pr_m  = tf.keras.metrics.AUC(curve="PR")
            auc_m.update_state(y_true, y_pred)
            pr_m.update_state(y_true, y_pred)
            auc, aupr = float(auc_m.result().numpy()), float(pr_m.result().numpy())
        res[name] = {"auc": auc, "aupr": aupr}

    valid_auc  = [v["auc"] for v in res.values() if v["auc"] is not None]
    valid_aupr = [v["aupr"] for v in res.values() if v["aupr"] is not None]
    summary = {
        "num_samples": int(X.shape[0]),
        "macro_auc": float(np.mean(valid_auc)) if valid_auc else None,
        "macro_aupr": float(np.mean(valid_aupr)) if valid_aupr else None,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_class": res}, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved → {args.out}")

if __name__ == "__main__":
    main()
