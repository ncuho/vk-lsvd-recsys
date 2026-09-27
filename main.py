# -*- coding: utf-8 -*-
"""
Влияние контентных эмбеддингов на рекомендации
================================================

Учебный проект: сравнение того, как добавление контентных признаков
(эмбеддингов и метаданных объектов) влияет на качество рекомендаций.

Датасет: VK-LSVD (https://huggingface.co/datasets/deepvk/VK-LSVD),
подвыборка up0.001_ip0.001 (0.1% пользователей и объектов).

Скрипт:
  1. Загружает взаимодействия (недели 0-24 — обучение, 25 — валидация,
     26 — тест) и строит разреженную матрицу «пользователь × объект».
  2. Загружает контентные признаки: 64-мерные эмбеддинги объектов и
     простые метаданные (author_id, duration).
  3. Реализует модели: Popularity, UserKNN, ItemKNN, MF (BPR),
     контентные профили (эмбеддинги / метаданные) и гибриды.
  4. Считает метрики Precision@10, Recall@10, NDCG@10, MAP@10.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.nearest_neighbours import CosineRecommender

# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

DATA_DIR = "data"
NEG_PER_USER = 100                # отрицательных кандидатов на пользователя
TOP_K = 10                        # по сколько элементов считаем метрики
MF_DIM = 64                       # латентная размерность матричной факторизации
MF_EPOCHS = 12                    # число эпох BPR
MF_LR = 0.05                      # скорость обучения
MF_REG = 0.01                     # L2-регуляризация
SEED = 42

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------

def load_interactions():
    """Читает недельные parquet-файлы. Возвращает train/val/test как
    pandas.DataFrame с колонками user_id, item_id."""
    weeks = {
        "train":      [f"week_{i:02d}.parquet" for i in range(0, 25)],
        "validation": ["week_25.parquet"],
        "test":       ["week_26.parquet"],
    }
    splits = {}
    for split, files in weeks.items():
        frames = [pd.read_parquet(os.path.join(DATA_DIR, "interactions", split, f),
                                  columns=["user_id", "item_id"])
                  for f in files]
        splits[split] = pd.concat(frames, ignore_index=True)
        print(f"[data] {split}: {len(splits[split]):,} взаимодействий")
    return splits


def build_ids(*id_series):
    """Сопоставляет большие исходные id с плотным диапазоном 0..N-1."""
    ids = np.unique(np.concatenate([s.values for s in id_series]))
    code = {int(x): i for i, x in enumerate(ids)}
    return code, ids


def to_matrix(df, user_code, item_code, n_users, n_items):
    """Строит бинарную разреженную матрицу user x item (1 — есть взаимодействие)."""
    u = df["user_id"].map(user_code).values.astype(np.int32)
    i = df["item_id"].map(item_code).values.astype(np.int32)
    data = np.ones(len(u), dtype=np.float32)
    return sp.csr_matrix((data, (u, i)), shape=(n_users, n_items))


# --------------------------------------------------------------------------
# Контентные признаки
# --------------------------------------------------------------------------

def load_embeddings(item_code):
    """Возвращает (n_items, 64) матрицу l2-нормированных эмбеддингов в
    порядке плотных индексов item_code."""
    path = os.path.join(DATA_DIR, "metadata", "item_embeddings.npz")
    z = np.load(path, mmap_mode="r")
    emb_item_id = np.asarray(z["item_id"])          # исходные id (полностью в памяти)
    lookup = pd.Series(np.arange(len(emb_item_id)), index=emb_item_id)
    idx = lookup.reindex(list(item_code.keys())).values   # индексы в порядке плотных id
    emb = z["embedding"][idx].astype(np.float32)    # читаем только нужные строки
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb


def load_metadata_features(item_code, n_items):
    """Строит контентный признак из метаданных: one-hot автора (top-200) +
    нормированная длительность. Возвращает (n_items, 201) l2-нормированную матрицу."""
    path = os.path.join(DATA_DIR, "metadata", "items_metadata.parquet")
    meta = pd.read_parquet(path, columns=["item_id", "author_id", "duration"])
    meta = meta.set_index("item_id").reindex(list(item_code.keys())).fillna(0)

    author = meta["author_id"].values.astype(np.int64)
    duration = meta["duration"].values.astype(np.float32)

    top_authors = pd.Series(author).value_counts().index[:200].tolist()
    author_code = {a: i for i, a in enumerate(top_authors)}
    n_a = len(top_authors) + 1                      # +1 корзина «прочие»

    feat = np.zeros((n_items, n_a), dtype=np.float32)
    for i, a in enumerate(author):
        feat[i, author_code.get(a, n_a - 1)] = 1.0

    dmin, dmax = duration.min(), duration.max()
    dnorm = (duration - dmin) / (dmax - dmin + 1e-9)
    feat = np.hstack([feat, dnorm[:, None]])
    feat /= (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    return feat.astype(np.float32)


# --------------------------------------------------------------------------
# Метрики ранжирования
# --------------------------------------------------------------------------

def dcg(scores):
    return float(np.sum((2.0 ** np.asarray(scores) - 1.0) /
                        np.log2(np.arange(2, len(scores) + 2))))


def ranking_metrics(ranked_relevances):
    """Precision@10, Recall@10, NDCG@10, MAP@10 по списку 0/1 релевантностей
    отсортированной выдачи (1 — релевантный объект)."""
    rel = np.asarray(ranked_relevances[:TOP_K], dtype=np.float64)
    n_rel = int(np.sum(np.asarray(ranked_relevances)))
    if n_rel == 0:
        return 0.0, 0.0, 0.0, 0.0

    hits = float(np.sum(rel))
    precision = hits / TOP_K
    recall = hits / n_rel
    ndcg = dcg(rel) / (dcg(np.ones(min(n_rel, TOP_K))) + 1e-9)

    correct = 0.0
    ap = 0.0
    for pos in range(1, TOP_K + 1):
        if rel[pos - 1] == 1:
            correct += 1
            ap += correct / pos
    map_k = ap / min(n_rel, TOP_K)
    return precision, recall, ndcg, map_k


def evaluate(model, user_positives, user_negatives):
    """Ранжирует для каждого пользователя его позитивные + негативные кандидаты
    и возвращает средние метрики."""
    prec, rec, ndcg, ap = [], [], [], []
    for u in user_positives:
        pos = user_positives[u]
        neg = user_negatives[u]
        candidates = list(pos) + list(neg)
        scores = np.asarray(model.score(u, candidates))
        order = np.argsort(-scores)
        ranked = np.array([1.0 if candidates[i] in pos else 0.0 for i in order])
        p, r, n, a = ranking_metrics(ranked)
        prec.append(p); rec.append(r); ndcg.append(n); ap.append(a)
    return (float(np.mean(prec)), float(np.mean(rec)),
            float(np.mean(ndcg)), float(np.mean(ap)))


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------

class Popularity:
    """Базовый уровень: скор объекта равен его популярности (числу взаимодействий)."""
    def __init__(self, train):
        self.pop = np.asarray(train.sum(axis=0)).ravel()

    def score(self, u, items):
        return self.pop[items]


class UserKNN:
    """User-based CF: скор = взвешенная сумма оценок похожих пользователей."""
    def __init__(self, train, k=100):
        self.train = train.astype(np.float32).tocsr()
        self.k = k
        # косинусное сходство пользователей (столбцы транспонированной матрицы)
        rec = CosineRecommender(K=k)
        rec.fit(self.train.T)
        self.sim = rec.similarity.tocsr()          # (n_users, n_users), top-k

    def score(self, u, items):
        s = self.sim.getrow(u) @ self.train        # (1, n_items)
        return np.asarray(s.toarray().ravel())[items]


class ItemKNN:
    """Item-based CF: скор = сумма сходств объекта с объектами пользователя."""
    def __init__(self, train, k=100):
        self.train = train.astype(np.float32).tocsr()
        self.k = k
        rec = CosineRecommender(K=k)
        rec.fit(self.train)
        self.sim = rec.similarity.tocsr()          # (n_items, n_items), top-k

    def score(self, u, items):
        interacted = self.train.indices[self.train.indptr[u]:self.train.indptr[u + 1]]
        if len(interacted) == 0:
            return np.zeros(len(items), dtype=np.float32)
        uv = np.zeros(self.train.shape[1], dtype=np.float32)
        uv[interacted] = 1.0
        return np.asarray(self.sim[items] @ uv).ravel()


class MF_BPR:
    """Матричная факторизация, обучаемая критерием BPR (pairwise-ранжирование)."""
    def __init__(self, train, dim=MF_DIM, epochs=MF_EPOCHS, lr=MF_LR, reg=MF_REG):
        self.train = train.astype(np.float32).tocsr()
        self.n_users, self.n_items = self.train.shape
        self.U = rng.normal(0, 0.1, (self.n_users, dim)).astype(np.float32)
        self.V = rng.normal(0, 0.1, (self.n_items, dim)).astype(np.float32)

        # плоский список (пользователь, позитивный объект) для сэмплирования троек
        pos_items, pos_owner = [], []
        for u in range(self.n_users):
            ps = self.train.indices[self.train.indptr[u]:self.train.indptr[u + 1]]
            if len(ps) > 0:
                pos_items.append(ps)
                pos_owner.append(np.full(len(ps), u, dtype=np.int32))
        self.pos_items = np.concatenate(pos_items)
        self.pos_owner = np.concatenate(pos_owner)
        self._fit(dim, epochs, lr, reg)

    def _fit(self, dim, epochs, lr, reg):
        steps, batch = 200, 2048
        for ep in range(epochs):
            loss = 0.0
            for _ in range(steps):
                idx = rng.integers(0, len(self.pos_items), batch)
                us = self.pos_owner[idx]
                pos = self.pos_items[idx]
                neg = rng.integers(0, self.n_items - 1, batch)
                neg = np.where(neg >= pos, neg + 1, neg)   # избегаем neg == pos

                up = self.U[us]
                vp = self.V[pos]
                vn = self.V[neg]
                x = np.sum(up * (vp - vn), axis=1)
                sig = 1.0 / (1.0 + np.exp(x))
                g = sig[:, None].astype(np.float32)

                np.add.at(self.U, us, lr * ((vp - vn) * g - reg * up))
                np.add.at(self.V, pos, lr * (up * g - reg * vp))
                np.add.at(self.V, neg, lr * (-up * g - reg * vn))
                loss += float(np.mean(np.log1p(np.exp(-x))))
            print(f"[mf] эпоха {ep + 1}/{epochs}, BPR loss={loss / steps:.4f}")

    def score(self, u, items):
        return np.asarray(self.U[u] @ self.V[items].T).ravel()


class ContentProfile:
    """Контентный профиль: пользователь = средний вектор объектов, с которыми
    он взаимодействовал; скор = близость объекта к этому профилю."""
    def __init__(self, train, item_features):
        self.train = train.astype(np.float32).tocsr()
        self.feats = item_features

    def score(self, u, items):
        interacted = self.train.indices[self.train.indptr[u]:self.train.indptr[u + 1]]
        if len(interacted) == 0:
            return np.zeros(len(items), dtype=np.float32)
        profile = self.feats[interacted].mean(axis=0)
        return np.asarray(self.feats[items] @ profile).ravel()


class Hybrid:
    """Гибрид: скор = (1-a)*коллаборативный + a*контентный (скоры z-нормируются)."""
    def __init__(self, collab, content, alpha=0.5):
        self.collab = collab
        self.content = content
        self.alpha = alpha

    def score(self, u, items):
        s1 = np.asarray(self.collab.score(u, items), dtype=np.float64)
        s2 = np.asarray(self.content.score(u, items), dtype=np.float64)
        s1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
        s2 = (s2 - s2.mean()) / (s2.std() + 1e-9)
        return (1 - self.alpha) * s1 + self.alpha * s2


# --------------------------------------------------------------------------
# Основной сценарий
# --------------------------------------------------------------------------

def main():
    splits = load_interactions()

    user_code, _ = build_ids(splits["train"]["user_id"], splits["validation"]["user_id"],
                             splits["test"]["user_id"])
    item_code, _ = build_ids(splits["train"]["item_id"], splits["validation"]["item_id"],
                             splits["test"]["item_id"])
    n_users, n_items = len(user_code), len(item_code)
    print(f"[data] пользователей: {n_users:,}, объектов: {n_items:,}")

    train = to_matrix(splits["train"], user_code, item_code, n_users, n_items)
    print(f"[data] плотность train: {train.nnz / (n_users * n_items):.4%}")

    # позитивные объекты теста для каждого пользователя (только «тёплые» — с историей)
    t0 = time.time()
    test_df = splits["test"]
    uc = test_df["user_id"].map(user_code).values.astype(np.int64)
    ic = test_df["item_id"].map(item_code).values.astype(np.int64)
    d = defaultdict(set)
    for u, i in zip(uc, ic):
        d[int(u)].add(int(i))
    user_positives = {u: s for u, s in d.items() if train[u].nnz > 0}
    print(f"[time] user_positives: {time.time() - t0:.1f}s ({len(user_positives)} пользователей)")

    # негативные кандидаты: смесь популярных и случайных объектов, не виденных пользователем
    t0 = time.time()
    pop = np.asarray(train.sum(axis=0)).ravel()
    popular_items = np.argsort(-pop)[:10_000]
    user_negatives = {}
    for u in user_positives:
        seen = set(train.indices[train.indptr[u]:train.indptr[u + 1]]) | user_positives[u]
        neg = set()
        # популярные негативы — берём из топ-200 популярных, не виденных пользователем
        for it in popular_items[:200]:
            if int(it) not in seen and len(neg) < NEG_PER_USER // 2:
                neg.add(int(it))
        # случайные негативы
        while len(neg) < NEG_PER_USER:
            it = int(rng.integers(0, n_items))
            if it not in seen:
                neg.add(it)
        user_negatives[u] = neg
    print(f"[time] negative sampling: {time.time() - t0:.1f}s")
    print(f"[data] оцениваемых пользователей: {len(user_positives):,}")

    t0 = time.time()
    emb = load_embeddings(item_code)
    print(f"[time] embeddings: {time.time() - t0:.1f}s")
    t0 = time.time()
    meta = load_metadata_features(item_code, n_items)
    print(f"[time] metadata: {time.time() - t0:.1f}s")
    print(f"[data] признаки: embeddings {emb.shape}, metadata {meta.shape}")

    models = {
        "Popularity": Popularity(train),
        "UserKNN": UserKNN(train),
        "ItemKNN": ItemKNN(train),
        "MF_BPR": MF_BPR(train),
        "ContentProfile_emb": ContentProfile(train, emb),
        "ContentProfile_meta": ContentProfile(train, meta),
    }
    models["Hybrid_ItemKNN+emb"] = Hybrid(models["ItemKNN"], models["ContentProfile_emb"], alpha=0.3)
    models["Hybrid_ItemKNN+meta"] = Hybrid(models["ItemKNN"], models["ContentProfile_meta"], alpha=0.3)
    models["Hybrid_MF+emb"] = Hybrid(models["MF_BPR"], models["ContentProfile_emb"], alpha=0.3)

    print("\n=== Результаты (Precision@10 / Recall@10 / NDCG@10 / MAP@10) ===")
    results = {}
    for name, model in models.items():
        p, r, n, a = evaluate(model, user_positives, user_negatives)
        results[name] = (p, r, n, a)
        print(f"{name:22s}  P@10={p:.4f}  R@10={r:.4f}  NDCG@10={n:.4f}  MAP@10={a:.4f}")

    out = pd.DataFrame(results, index=["Precision@10", "Recall@10", "NDCG@10", "MAP@10"]).T
    out.to_csv(os.path.join(DATA_DIR, "results.csv"))
    print("\nСохранено:", os.path.join(DATA_DIR, "results.csv"))


if __name__ == "__main__":
    main()
