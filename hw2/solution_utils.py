"""HW2: функции, которые импортируют solution.ipynb и eda.ipynb.

Задача одна: реакция пользователя (positive / neutral / negative), как в задаче 1 HW1.
Модели делятся на две группы по префиксу имени:
  hw1_ -- модели HW1 без изменений (3 базовые и своя по репликам);
  hw2_ -- новые: случайный лес, два градиентных бустинга, стекинг и блендинг.
Используются только методы занятий 4, 8 и 9: TF-IDF, Pipeline, ColumnTransformer, группы в
кросс-валидации, joblib, деревья, случайный лес, важность признаков, градиентный бустинг,
библиотеки бустинга, предсказания вне блоков (out-of-fold), стекинг и блендинг.
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
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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


def read_labels(path: str, task: str = "reaction") -> dict[str, str]:
    """Метки из поля labels (как в train) или с верхнего уровня записи (формат hidden_labels.jsonl)."""
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or record).get(task)
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
#  Нормализация текста (занятие 4)
# ---------------------------------------------------------------------------

def normalize(text: str) -> list[str]:
    """Нижний регистр, слова из букв, язык по алфавиту, без стоп-слов, основы слов."""
    words = WORD.findall(text.lower())
    letters = "".join(words)
    language = "russian" if letters and len(CYRILLIC.findall(letters)) > len(letters) / 2 else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    return [stemmer.stem(w) for w in words if len(w) >= 2 and w not in stop]


# ---------------------------------------------------------------------------
#  Признаки диалога: тексты, структура
# ---------------------------------------------------------------------------

PY_KW = re.compile(r"\b(def|class|import|return|lambda|None|self)\b")
REFUSAL_PHRASES = ("I'm sorry", "I can't", "I cannot", "I am unable", "I'm unable")

# Структурные признаки HW1 (из блока 8 EDA) и полный список, который был в HW1.
NUM_COLS = ["first_words", "last_words", "has_code", "has_py_kw", "has_traceback"]
NUM_COLS_ALL = ["first_words", "last_words", "n_user_turns",
                "has_code", "has_traceback", "has_refusal", "has_py_kw"]
# Дополнительные признаки для деревьев: длины, знаки в последней реплике, блоки кода.
EXTRA_COLS = ["asst_words", "user_words", "max_user_words", "last_q", "last_excl", "last_upper", "code_blocks"]
TREE_NUM_COLS = NUM_COLS_ALL + EXTRA_COLS


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


def extract_struct(dialog: dict) -> dict[str, float]:
    turns = dialog["turns"]
    user_turns = [t["text"] for t in turns if t["role"] == "user"]
    asst_joined = " ".join(t["text"] for t in turns if t["role"] == "assistant")
    first_user = user_turns[0] if user_turns else ""
    last_user = user_turns[-1] if user_turns else ""
    letters = [c for c in last_user if c.isalpha()]
    user_lens = [len(t.split()) for t in user_turns] or [0]
    return {
        "first_words": len(first_user.split()),
        "last_words": len(last_user.split()),
        "n_user_turns": len(user_turns),
        "has_code": int("```" in asst_joined),
        "has_traceback": int(("Traceback" in asst_joined) or ("Error:" in asst_joined)),
        "has_refusal": int(any(s in asst_joined for s in REFUSAL_PHRASES)),
        "has_py_kw": int(bool(PY_KW.search(asst_joined))),
        "asst_words": len(asst_joined.split()),
        "user_words": sum(user_lens),
        "max_user_words": max(user_lens),
        "last_q": last_user.count("?"),
        "last_excl": last_user.count("!"),
        "last_upper": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "code_blocks": asst_joined.count("```") // 2,
    }


def features_frame(dialogs: list[dict]) -> pd.DataFrame:
    """Одна строка на диалог: три нормализованных текста и числовые признаки."""
    return pd.DataFrame([{**extract_texts(d), **extract_struct(d)} for d in dialogs])


_CACHE: dict = {}


def _cached(kind: str, dialogs: list, builder):
    """Признаки считаются один раз на список диалогов (много моделей -> один расчет)."""
    entry = _CACHE.get(kind)
    if entry is not None and entry[0] is dialogs and entry[1] == len(dialogs):
        return entry[2]
    value = builder()
    _CACHE[kind] = (dialogs, len(dialogs), value)
    return value


# ---------------------------------------------------------------------------
#  Группы для кросс-валидации (занятие 4, задача 14 лабораторной)
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
    """Разбиение по группам, как в HW1: группы перемешиваются (rng.permutation), группа на месте j
    попадает в часть j mod n_splits вместе со всеми объектами. Возвращает пары (обучение, проверка)."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(groups))
    part_of = {g: j % n_splits for j, g in enumerate(shuffled)}
    part = np.array([part_of[g] for g in groups])
    return [(np.where(part != f)[0], np.where(part == f)[0]) for f in range(n_splits)]


# ---------------------------------------------------------------------------
#  Метрики под сбалансированный скрытый тест (как в HW1)
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
        out["recall_" + c[:3]] = float(r)
        out["prec_bal_" + c[:3]] = float(p)
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
#  Модели HW1 (без изменений): 3 базовые и своя по репликам
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
    """hw1_base_logreg: LogReg на 3 TF-IDF + структура."""
    num_cols = NUM_COLS if num_cols is None else num_cols
    return Pipeline([("prep", _dialog_preprocessor(num_cols)),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight))])


def build_nb_pipeline() -> Pipeline:
    """hw1_base_nb: MultinomialNB на тексте пользователя."""
    return Pipeline([("tfidf", _tfidf(3, 30000)), ("clf", MultinomialNB(alpha=0.1))])


def build_svc_pipeline(num_cols=None, class_weight=None) -> Pipeline:
    """hw1_base_svm: LinearSVC на 3 TF-IDF + структура."""
    num_cols = NUM_COLS if num_cols is None else num_cols
    return Pipeline([("prep", _dialog_preprocessor(num_cols)),
                     ("clf", LinearSVC(C=1.0, max_iter=3000, dual="auto", class_weight=class_weight))])


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
    """hw1_custom_turn_h8: классификатор реакции отдельной реплики. Реакция диалога = реакция
    последней не-neutral реплики (правило H8, совпало с разметкой в 3000 из 3000).
    thr_neg / thr_pos -- пороги вероятности, подобранные в HW1."""

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


TURN_AGG_COLS = ["tp_neg_max", "tp_pos_max", "tp_neg_mean", "tp_pos_mean", "tp_neg_last", "tp_pos_last",
                 "tp_neg_cnt", "tp_pos_cnt", "h8_neg", "h8_pos"]

NEW_CACHE = {}


def cached_features(dialogs):
    entry = NEW_CACHE.get("feat")
    if entry is not None:
        if entry[0] is dialogs and entry[1] == len(dialogs):
            return entry[2]
    value = features_frame(dialogs)
    NEW_CACHE["feat"] = (dialogs, len(dialogs), value)
    return value


def cached_turns(dialogs):
    entry = NEW_CACHE.get("turn")
    if entry is not None:
        if entry[0] is dialogs and entry[1] == len(dialogs):
            return entry[2]
    value = turn_frame(dialogs)
    NEW_CACHE["turn"] = (dialogs, len(dialogs), value)
    return value


def turn_proba_oof(frame, y_turn, splits):
    dlg = frame["dialog_idx"].values
    y_turn = np.asarray(y_turn)
    P = np.zeros((len(frame), 3))
    for tr, va in splits:
        tr_rows = np.where(np.isin(dlg, tr))[0]
        va_rows = np.where(np.isin(dlg, va))[0]
        model = TurnReactionModel().fit_frame(frame.iloc[tr_rows], y_turn[tr_rows])
        P[va_rows] = model.proba_frame(frame.iloc[va_rows])
    return P


def turn_aggregates(frame, proba, n_dialogs, turn_model):
    d = pd.DataFrame({"dialog_idx": frame["dialog_idx"].values, "p_neg": proba[:, 0], "p_pos": proba[:, 2]})
    d["neg_high"] = (d["p_neg"] > 0.5).astype(int)
    d["pos_high"] = (d["p_pos"] > 0.5).astype(int)
    g = d.groupby("dialog_idx")
    agg = pd.DataFrame({
        "tp_neg_max": g["p_neg"].max(), "tp_pos_max": g["p_pos"].max(),
        "tp_neg_mean": g["p_neg"].mean(), "tp_pos_mean": g["p_pos"].mean(),
        "tp_neg_last": g["p_neg"].last(), "tp_pos_last": g["p_pos"].last(),
        "tp_neg_cnt": g["neg_high"].sum(), "tp_pos_cnt": g["pos_high"].sum(),
    })
    h8 = turn_model.decide(frame, proba)
    h8_neg = []
    h8_pos = []
    for i in agg.index:
        h8_neg.append(int(h8.get(i) == "negative"))
        h8_pos.append(int(h8.get(i) == "positive"))
    agg["h8_neg"] = h8_neg
    agg["h8_pos"] = h8_pos
    agg = agg.reindex(range(n_dialogs)).fillna(0.0)
    return agg[TURN_AGG_COLS]


def reaction_frame(dialogs, turn_model=None):
    F = cached_features(dialogs)
    if turn_model is None:
        return F
    TF = cached_turns(dialogs)
    agg = turn_aggregates(TF, turn_model.proba_frame(TF), len(dialogs), turn_model)
    return pd.concat([F, agg], axis=1)


def tree_prep(num_cols, n_words):
    parts = []
    if n_words > 0:
        parts.append(("tail", TfidfVectorizer(max_features=n_words, min_df=5, max_df=0.9, sublinear_tf=True), "text_tail"))
        parts.append(("user", TfidfVectorizer(max_features=n_words, min_df=5, max_df=0.9, sublinear_tf=True), "text_user"))
    parts.append(("num", "passthrough", list(num_cols)))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)


def build_forest(num_cols, n_words=300, n_estimators=300, max_features="sqrt", min_samples_leaf=2):
    forest = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                    min_samples_leaf=min_samples_leaf, class_weight="balanced",
                                    n_jobs=-1, random_state=0)
    return Pipeline([("prep", tree_prep(num_cols, n_words)), ("clf", forest)])


def build_hist_gb(num_cols, n_words=300, learning_rate=0.1, max_depth=4, max_iter=300, early_stopping=True):
    boosting = HistGradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, max_iter=max_iter,
                                              class_weight="balanced", early_stopping=early_stopping,
                                              validation_fraction=0.15, n_iter_no_change=20, random_state=0)
    return Pipeline([("prep", tree_prep(num_cols, n_words)), ("clf", boosting)])


def build_lgbm(num_cols, n_words=300, n_estimators=300, learning_rate=0.05, num_leaves=15):
    boosting = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              class_weight="balanced", random_state=0, n_jobs=-1, verbose=-1)
    return Pipeline([("prep", tree_prep(num_cols, n_words)), ("clf", boosting)])


class TreeModel:
    def __init__(self, pipe, turn_model=None):
        self.pipe = pipe
        self.turn_model = turn_model

    def predict_dialogs(self, dialogs):
        X = reaction_frame(dialogs, self.turn_model)
        proba = self.pipe.predict_proba(X)
        answers = []
        for i in proba.argmax(axis=1):
            answers.append(CLASSES[i])
        return answers


class StackModel:
    def __init__(self, members, meta=None, turn_model=None):
        self.members = members
        self.meta = meta
        self.turn_model = turn_model

    def predict_dialogs(self, dialogs):
        X = reaction_frame(dialogs, self.turn_model)
        parts = []
        for pipe, how in self.members:
            if how == "text":
                parts.append(pipe.predict_proba(X["text_user"]))
            elif how == "decision":
                parts.append(pipe.decision_function(X))
            else:
                parts.append(pipe.predict_proba(X))
        if self.meta is None:
            scores = np.mean(parts, axis=0)
        else:
            scores = self.meta.predict_proba(np.hstack(parts))
        answers = []
        for i in scores.argmax(axis=1):
            answers.append(CLASSES[i])
        return answers


OLD_MODEL_NAMES = ["hw1_base_logreg", "hw1_base_nb", "hw1_base_svm", "hw1_custom_turn_h8"]
NEW_MODEL_NAMES = ["hw2_random_forest", "hw2_hist_gb", "hw2_lightgbm", "hw2_stacking", "hw2_blend"]
REACTION_MODEL_NAMES = OLD_MODEL_NAMES + NEW_MODEL_NAMES


def save_reaction_models(models, art_dir=ARTIFACTS_DIR):
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        joblib.dump(models[name], art_dir / (name + ".joblib"), compress=3)


def init_reaction_models(train_path=None, art_dir=ARTIFACTS_DIR):
    art_dir = Path(art_dir)
    models = {}
    for name in REACTION_MODEL_NAMES:
        path = art_dir / (name + ".joblib")
        if path.exists():
            models[name] = joblib.load(path)
    if len(models) == 0:
        raise FileNotFoundError("В папке " + str(art_dir) + " нет файлов hw1_*.joblib и hw2_*.joblib, сначала запустите eda.ipynb")
    return models


def predict_reaction(model, dialogs):
    if hasattr(model, "predict_dialogs"):
        return model.predict_dialogs(dialogs)
    df = cached_features(dialogs)
    first_step = list(model.named_steps.values())[0]
    if isinstance(first_step, TfidfVectorizer):
        X = df["text_user"]
    else:
        X = df
    answers = []
    for p in model.predict(X):
        answers.append(str(p))
    return answers


def save_predictions(path, dialogs, predictions):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for name in predictions:
            answers = predictions[name]
            for i in range(len(dialogs)):
                writer.writerow([dialogs[i]["dialog_id"], "reaction", name, answers[i]])
