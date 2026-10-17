"""HW1: функции, которые импортируют solution.ipynb и eda.ipynb.

Задача 1 (реакция): признаки, 4 модели, метрики под сбалансированный тест,
сохранение/загрузка моделей из artifacts/.
Задача 2 (темы): TF-IDF -> PCA -> кластеризация -> классификатор.
Используются только методы с занятий (TF-IDF, Pipeline, ColumnTransformer, LogReg, NB, SVM, class_weight,
порог по predict_proba, разбиение по группам из лабораторной, PCA, k-средних, Ward, GMM, joblib).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score, completeness_score, confusion_matrix,
    f1_score, homogeneity_score, normalized_mutual_info_score, v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CLASSES = ["negative", "neutral", "positive"]

# Чистка: длинные вставки (код, копипаста) обрезаются до этих длин в символах.
MAX_USER_CHARS = 2000
MAX_ASST_CHARS = 800


# ---------------------------------------------------------------------------
#  Данные
# ---------------------------------------------------------------------------

def read_dialogs(path: str) -> list[dict]:
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def read_turn_labels(path: str) -> dict[str, list[str]]:
    """dialog_id -> реакция каждой user-реплики (None у первой реплики = neutral)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                user_turns = (record.get("labels") or {}).get("user_turns")
                if user_turns:
                    out[record["dialog_id"]] = [(u.get("reaction") or "neutral") for u in user_turns]
    return out


# ---------------------------------------------------------------------------
#  Нормализация текста
# ---------------------------------------------------------------------------

def normalize(text: str) -> list[str]:
    """Нормализация с семинара 4: слова, язык по алфавиту, стоп-слова, стемминг."""
    words = WORD.findall(text.lower())
    letters = "".join(words)
    language = "russian" if letters and len(CYRILLIC.findall(letters)) > len(letters) / 2 else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    return [stemmer.stem(w) for w in words if len(w) >= 2 and w not in stop]


def dialog_text(dialog: dict) -> str:
    """Для заглушки задачи 2 (как в примере)."""
    user_text = " ".join(t["text"] for t in dialog["turns"] if t["role"] == "user")
    return " ".join(normalize(user_text))


# ---------------------------------------------------------------------------
#  Признаки диалога
# ---------------------------------------------------------------------------

PY_KW = re.compile(r"\b(def|class|import|return|lambda|None|self)\b")
REFUSAL_PHRASES = ("I'm sorry", "I can't", "I cannot", "I am unable", "I'm unable")

NUM_COLS_ALL = ["first_words", "last_words", "n_user_turns",
                "has_code", "has_traceback", "has_refusal", "has_py_kw"]
# Отобрано в блоке 8 EDA (там был сигнал); абляция в «Задаче 1» проверяет, нужны ли они моделям.
NUM_COLS = ["first_words", "last_words", "has_code", "has_py_kw", "has_traceback"]


def _last_user_with_context(turns: list[dict]) -> str:
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user":
            prev_asst = ""
            if i >= 1 and turns[i - 1]["role"] == "assistant":
                prev_asst = turns[i - 1]["text"][:MAX_ASST_CHARS]
            return (prev_asst + " " + turns[i]["text"][:MAX_USER_CHARS]).strip()
    return ""


def extract_texts(dialog: dict) -> dict[str, str]:
    turns = dialog["turns"]
    user_text = " ".join(t["text"][:MAX_USER_CHARS] for t in turns if t["role"] == "user")
    asst_text = " ".join(t["text"][:MAX_ASST_CHARS] for t in turns if t["role"] == "assistant")
    return {
        "text_user": " ".join(normalize(user_text)),
        "text_asst": " ".join(normalize(asst_text)),
        "text_tail": " ".join(normalize(_last_user_with_context(turns))),
    }


def extract_struct(dialog: dict) -> dict[str, int]:
    turns = dialog["turns"]
    user_turns = [t for t in turns if t["role"] == "user"]
    asst_joined = " ".join(t["text"] for t in turns if t["role"] == "assistant")
    first_user = user_turns[0]["text"] if user_turns else ""
    last_user = user_turns[-1]["text"] if user_turns else ""
    return {
        "first_words": len(first_user.split()),
        "last_words": len(last_user.split()),
        "n_user_turns": len(user_turns),
        "has_code": int("```" in asst_joined),
        "has_traceback": int(("Traceback" in asst_joined) or ("Error:" in asst_joined)),
        "has_refusal": int(any(s in asst_joined for s in REFUSAL_PHRASES)),
        "has_py_kw": int(bool(PY_KW.search(asst_joined))),
    }


def dialog_features(dialog: dict) -> dict:
    return {**extract_texts(dialog), **extract_struct(dialog)}


_CACHE: dict = {}


def _cached(kind: str, dialogs: list, builder):
    """Признаки считаются один раз на список диалогов (4 модели -> 1 расчет)."""
    entry = _CACHE.get(kind)
    if entry is not None and entry[0] is dialogs and entry[1] == len(dialogs):
        return entry[2]
    value = builder()
    _CACHE[kind] = (dialogs, len(dialogs), value)
    return value


def features_frame(dialogs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([dialog_features(d) for d in dialogs])


# ---------------------------------------------------------------------------
#  Группы для кросс-валидации (задачи 14 и 15 лабораторной)
# ---------------------------------------------------------------------------

def first_message_key(dialog: dict) -> str:
    """Ключ группы: одинаковая первая реплика пользователя (без регистра и лишних пробелов)."""
    first = next((t["text"] for t in dialog["turns"] if t["role"] == "user"), "")
    return " ".join(first.lower().split())[:200]


def group_ids(dialogs: list[dict]) -> np.ndarray:
    """Номер группы каждого диалога: диалоги с одной и той же первой репликой -- одна группа."""
    _, inverse = np.unique([first_message_key(d) for d in dialogs], return_inverse=True)
    return inverse


def group_kfold_splits(groups, n_splits: int = 5, seed: int = 0):
    """Разбиение по группам, как в задаче 14 лабораторной: группы перемешиваются
    (rng.permutation), группа на месте j попадает в часть j mod n_splits вместе со всеми объектами.
    Возвращает пары (индексы обучения, индексы проверки)."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(groups))
    part_of = {g: j % n_splits for j, g in enumerate(shuffled)}
    part = np.array([part_of[g] for g in groups])
    return [(np.where(part != f)[0], np.where(part == f)[0]) for f in range(n_splits)]


# ---------------------------------------------------------------------------
#  Метрики под сбалансированный скрытый тест
# ---------------------------------------------------------------------------

def balanced_report(y_true, y_pred) -> dict[str, float]:
    """Метрики так, как если бы классов в проверке было поровну (как на скрытом тесте).

    Матрицу ошибок нормируем по строкам: M[i, j] = доля объектов класса i, предсказанных как j
    (диагональ = recall). При равных долях классов precision_j = M[j, j] / sum_i M[i, j].
    Из них считаем F1 каждого класса и macro-F1 (f1_macro_bal). bal_acc = средний recall.
    f1_macro_raw -- обычный macro-F1 на данных как есть."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).astype(float)
    row = cm.sum(axis=1, keepdims=True)
    M = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    recall = np.diag(M)
    col = M.sum(axis=0)
    precision = np.divide(recall, col, out=np.zeros_like(recall), where=col > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    out = {
        "f1_macro_bal": float(f1.mean()),
        "bal_acc": float(recall.mean()),
        "f1_macro_raw": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
    }
    for c, r, p in zip(CLASSES, recall, precision):
        out[f"recall_{c[:3]}"] = float(r)
        out[f"prec_bal_{c[:3]}"] = float(p)
    return out


def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
#  Модели задачи 1 на уровне диалога
# ---------------------------------------------------------------------------

def _tfidf(min_df, max_features):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.9,
                           sublinear_tf=True, max_features=max_features)


def _dialog_preprocessor(num_cols):
    transformers = [
        ("user", _tfidf(3, 30000), "text_user"),
        ("asst", _tfidf(3, 30000), "text_asst"),
        ("tail", _tfidf(2, 20000), "text_tail"),
    ]
    if num_cols:
        transformers.append(("num", StandardScaler(), list(num_cols)))
    return ColumnTransformer(transformers, remainder="drop")


def build_reaction_pipeline(num_cols=None, class_weight=None) -> Pipeline:
    """Базовая 1: LogReg на 3 TF-IDF + структура."""
    num_cols = NUM_COLS if num_cols is None else num_cols
    return Pipeline([("prep", _dialog_preprocessor(num_cols)),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight))])


def build_nb_pipeline() -> Pipeline:
    """Базовая 2: MultinomialNB на тексте пользователя."""
    return Pipeline([("tfidf", _tfidf(3, 30000)),
                     ("clf", MultinomialNB(alpha=0.1))])


def build_svc_pipeline(num_cols=None, class_weight=None) -> Pipeline:
    """Базовая 3: LinearSVC на 3 TF-IDF + структура."""
    num_cols = NUM_COLS if num_cols is None else num_cols
    return Pipeline([("prep", _dialog_preprocessor(num_cols)),
                     ("clf", LinearSVC(C=1.0, max_iter=3000, dual="auto", class_weight=class_weight))])


def build_text_pipeline(column: str, class_weight=None) -> Pipeline:
    """Один TF-IDF по одной колонке (абляция: какая часть диалога несет сигнал)."""
    prep = ColumnTransformer([("t", _tfidf(2, 30000), column)], remainder="drop")
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight))])


# ---------------------------------------------------------------------------
#  Своя модель: реакция каждой реплики + правило H8
# ---------------------------------------------------------------------------

TURN_NUM_COLS = ["is_first", "is_last", "words", "prev_words"]


def turn_frame(dialogs: list[dict]) -> pd.DataFrame:
    """Одна строка = одна user-реплика + ответ ассистента перед ней."""
    rows = []
    for di, d in enumerate(dialogs):
        turns = d["turns"]
        n_user = sum(t["role"] == "user" for t in turns)
        prev, k = "", 0
        for t in turns:
            if t["role"] == "assistant":
                prev = t["text"]
                continue
            rows.append({
                "dialog_idx": di, "turn_idx": k,
                "text_turn": " ".join(normalize(t["text"][:MAX_USER_CHARS])),
                "text_prev": " ".join(normalize(prev[:MAX_ASST_CHARS])),
                "is_first": int(k == 0), "is_last": int(k == n_user - 1),
                "words": float(np.log1p(len(t["text"].split()))),
                "prev_words": float(np.log1p(len(prev.split()))),
            })
            k += 1
            prev = ""
    return pd.DataFrame(rows)


def turn_targets(frame: pd.DataFrame, dialogs: list[dict], turn_labels: dict[str, list[str]]) -> np.ndarray:
    y = []
    for di, k in zip(frame["dialog_idx"].values, frame["turn_idx"].values):
        lst = turn_labels.get(dialogs[di]["dialog_id"], [])
        y.append(lst[k] if k < len(lst) else "neutral")
    return np.array(y)


class TurnReactionModel:
    """Классификатор реакции отдельной реплики. Реакция диалога = реакция
    последней не-neutral реплики (правило H8, совпало с разметкой в 3000 из 3000).
    thr_neg / thr_pos -- пороги вероятности; подбираются по OOF в eda.ipynb."""

    def __init__(self, thr_neg: float = 0.5, thr_pos: float = 0.5, C: float = 1.0):
        self.thr_neg, self.thr_pos, self.C = thr_neg, thr_pos, C
        self.pipe = None

    def _build(self) -> Pipeline:
        prep = ColumnTransformer([
            ("turn", _tfidf(2, 30000), "text_turn"),
            ("prev", _tfidf(3, 20000), "text_prev"),
            ("num", StandardScaler(), TURN_NUM_COLS),
        ], remainder="drop")
        clf = LogisticRegression(max_iter=2000, C=self.C, class_weight="balanced")
        return Pipeline([("prep", prep), ("clf", clf)])

    def fit_frame(self, frame: pd.DataFrame, y_turn) -> "TurnReactionModel":
        self.pipe = self._build().fit(frame, np.asarray(y_turn))
        return self

    def proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Вероятности в порядке CLASSES."""
        proba = self.pipe.predict_proba(frame)
        order = [list(self.pipe.classes_).index(c) for c in CLASSES]
        return proba[:, order]

    def decide(self, frame: pd.DataFrame, proba: np.ndarray, thr_neg=None, thr_pos=None) -> dict[int, str]:
        tn = self.thr_neg if thr_neg is None else thr_neg
        tp = self.thr_pos if thr_pos is None else thr_pos
        pn, pp = proba[:, 0], proba[:, 2]
        cand = np.where((pn >= tn) & (pn >= pp), "negative",
                        np.where((pp >= tp) & (pp > pn), "positive", "neutral"))
        out: dict[int, str] = {}
        for di, c in zip(frame["dialog_idx"].values, cand):
            out.setdefault(int(di), "neutral")
            if c != "neutral":
                out[int(di)] = c  # строки идут по порядку реплик -> останется последняя не-neutral
        return out

    def predict_dialogs(self, dialogs: list[dict]) -> list[str]:
        frame = _cached("turn", dialogs, lambda: turn_frame(dialogs))
        res = self.decide(frame, self.proba_frame(frame))
        return [res.get(i, "neutral") for i in range(len(dialogs))]


# ---------------------------------------------------------------------------
#  Обучение, сохранение, загрузка
# ---------------------------------------------------------------------------

REACTION_MODEL_NAMES = ["base_logreg", "base_nb", "base_svm", "custom_turn_h8"]


def fit_reaction_model(name: str, df: pd.DataFrame, y, turn_fr=None, y_turn=None, turn_params: dict | None = None):
    """Обучить одну модель. df -- features_frame, turn_fr/y_turn -- для custom_turn_h8.
    Дисбаланс классов: class_weight="balanced" (тест сбалансирован, а в train neutral больше половины)."""
    y = np.asarray(y)
    if name == "custom_turn_h8":
        return TurnReactionModel(**(turn_params or {})).fit_frame(turn_fr, y_turn)
    if name == "base_nb":
        return build_nb_pipeline().fit(df["text_user"], y)
    builders = {"base_logreg": build_reaction_pipeline, "base_svm": build_svc_pipeline}
    return builders[name](class_weight="balanced").fit(df, y)


def save_reaction_models(models: dict, art_dir=ARTIFACTS_DIR) -> None:
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, art_dir / f"reaction_{name}.joblib", compress=3)


def init_reaction_models(train_path: str | None = None, art_dir=ARTIFACTS_DIR) -> dict:
    """Загрузка обученных моделей из artifacts (train_path не нужен, оставлен для совместимости)."""
    art_dir = Path(art_dir)
    models = {}
    for name in REACTION_MODEL_NAMES:
        path = art_dir / f"reaction_{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    if not models:
        raise FileNotFoundError(f"В {art_dir} нет файлов reaction_*.joblib: сначала запустите eda.ipynb")
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    if isinstance(model, TurnReactionModel):
        return model.predict_dialogs(dialogs)
    df = _cached("feat", dialogs, lambda: features_frame(dialogs))
    first_step = list(model.named_steps.values())[0]
    X = df["text_user"] if isinstance(first_step, TfidfVectorizer) else df
    return [str(p) for p in model.predict(X)]


# ---------------------------------------------------------------------------
#  Задача 2: темы (TF-IDF -> PCA -> кластеризация -> классификатор)
# ---------------------------------------------------------------------------

MAX_TOPIC_CHARS = 6000


def topic_texts(dialogs: list[dict]) -> list[str]:
    """Текст диалога для тем: первая user-реплика (она обычно задает тему) стоит дважды,
    чтобы ее слова весили больше, затем все user-реплики."""
    out = []
    for d in dialogs:
        user = [t["text"][:MAX_USER_CHARS] for t in d["turns"] if t["role"] == "user"]
        first = " ".join(normalize(user[0])) if user else ""
        rest = " ".join(normalize(" ".join(user)[:MAX_TOPIC_CHARS]))
        out.append(f"{first} {first} {rest}")
    return out


class TopicEmbedder:
    """TF-IDF (разреженный, как на семинаре 4) -> PCA (занятие 5; на плотной матрице, поэтому
    словарь ограничен max_features слов)."""

    def __init__(self, n_components: int = 100, max_features: int = 3000, random_state: int = 0):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5,
                                     sublinear_tf=True, max_features=max_features)
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, texts: list[str]) -> "TopicEmbedder":
        self.pca.fit(self.tfidf.fit_transform(texts).toarray())
        return self

    def sparse(self, texts: list[str]):
        return self.tfidf.transform(texts)

    def dense(self, S) -> np.ndarray:
        return self.pca.transform(S.toarray())


def fit_topic_clusterer(kind: str, E: np.ndarray, k: int, seed: int = 0, ward_sample: int = 4000):
    """Кластеризация эмбеддингов. Возвращает (оценщик с predict, метки всех объектов).
    kmeans -- k-средних; gmm -- смесь гауссиан; ward_logreg -- Ward на подвыборке
    (на всех данных не хватит памяти) + логистическая регрессия, переносящая метки на все диалоги."""
    if kind == "kmeans":
        est = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(E)
        return est, est.labels_
    if kind == "gmm":
        est = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, max_iter=200).fit(E)
        return est, est.predict(E)
    if kind == "ward_logreg":
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(E), size=min(ward_sample, len(E)), replace=False)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(E[sub])
        est = LogisticRegression(max_iter=2000, C=10.0).fit(E[sub], ag.labels_)
        return est, est.predict(E)
    raise ValueError(kind)


TOPIC_CLASSIFIERS = ["base_logreg", "base_nb", "base_svm", "custom_svc_rbf"]


def fit_topic_classifier(kind: str, E, S, y):
    """Классификатор тем, обученный на метках кластеризации. Возвращает (оценщик, пространство).
    base_logreg -- LogReg на PCA-эмбеддингах; base_nb и base_svm -- MultinomialNB и LinearSVC на TF-IDF;
    custom_svc_rbf -- своя: SVC с ядром RBF (ядровой переход, занятие 5) на PCA-эмбеддингах."""
    if kind == "base_logreg":
        return LogisticRegression(max_iter=2000, C=10.0).fit(E, y), "emb"
    if kind == "base_nb":
        return MultinomialNB(alpha=0.1).fit(S, y), "sparse"
    if kind == "base_svm":
        return LinearSVC(C=0.5, dual="auto").fit(S, y), "sparse"
    if kind == "custom_svc_rbf":
        return SVC(kernel="rbf", C=10.0, gamma="scale").fit(E, y), "emb"
    raise ValueError(kind)


class TopicModel:
    """Оценщик + эмбеддер + названия тем. Применяется к списку диалогов."""

    def __init__(self, embedder: TopicEmbedder, estimator, names: list[str], space: str = "emb"):
        self.embedder, self.estimator, self.names, self.space = embedder, estimator, list(names), space

    def predict_dialogs(self, dialogs: list[dict]) -> list[str]:
        texts = _cached("topic", dialogs, lambda: topic_texts(dialogs))
        S = self.embedder.sparse(texts)
        X = S if self.space == "sparse" else self.embedder.dense(S)
        return [self.names[int(c)] for c in self.estimator.predict(X)]


def cluster_top_terms(S, labels, feature_names, k: int, topn: int = 12) -> dict[int, list[str]]:
    """Характерные слова кластера: средний TF-IDF внутри кластера минус средний по всем."""
    labels = np.asarray(labels)
    overall = np.asarray(S.mean(axis=0)).ravel()
    out = {}
    for c in range(k):
        rows = np.where(labels == c)[0]
        if len(rows) == 0:
            out[c] = []
            continue
        diff = np.asarray(S[rows].mean(axis=0)).ravel() - overall
        out[c] = [feature_names[i] for i in np.argsort(-diff)[:topn]]
    return out


def closest_examples(E: np.ndarray, labels, k: int, n: int = 3) -> dict[int, np.ndarray]:
    """Индексы n диалогов, ближайших к центру каждого кластера (характерные диалоги)."""
    labels = np.asarray(labels)
    out = {}
    for c in range(k):
        rows = np.where(labels == c)[0]
        if len(rows) == 0:
            out[c] = np.array([], dtype=int)
            continue
        center = E[rows].mean(axis=0)
        out[c] = rows[np.argsort(((E[rows] - center) ** 2).sum(axis=1))[:n]]
    return out


TOPIC_MODEL_NAMES = list(TOPIC_CLASSIFIERS)


def save_topic_models(models: dict, art_dir=ARTIFACTS_DIR) -> None:
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, art_dir / f"topic_{name}.joblib", compress=3)


def init_topic_models(train_path: str | None = None, art_dir=ARTIFACTS_DIR) -> dict:
    """Загрузка моделей тем из artifacts; если их нет -- заглушка из примера."""
    art_dir = Path(art_dir)
    models = {}
    for name in TOPIC_MODEL_NAMES:
        path = art_dir / f"topic_{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    if models:
        return models
    dialogs = read_dialogs(train_path)
    ids = np.array([d["dialog_id"] for d in dialogs]).reshape(-1, 1)
    return {"dummy_one_topic": DummyClassifier(strategy="most_frequent").fit(ids, ["all"] * len(dialogs))}


def predict_topic(model, dialogs: list[dict]) -> list[str]:
    if isinstance(model, TopicModel):
        return model.predict_dialogs(dialogs)
    ids = np.array([d["dialog_id"] for d in dialogs]).reshape(-1, 1)
    return [str(label) for label in model.predict(ids)]


def compute_partition_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """ARI, NMI, однородность, полнота, V-мера, чистота (как в readme)."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    purity = pd.crosstab(pd.Series(y_pred), pd.Series(y_true)).max(axis=1).sum() / max(len(pairs), 1)
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
        "purity": float(purity),
    }


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])
