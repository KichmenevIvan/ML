"""Пример решения первой домашней работы: функции, которые импортирует solution.ipynb.

В примере вместо моделей стоит DummyClassifier: он не смотрит на текст
и показывает, какое качество получается совсем без модели. В вашем решении
функции init_*_models загружают обученные модели из папки artifacts,
а predict_* делают ту обработку текста, на которой модели обучались.
"""

from __future__ import annotations

import csv
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}


# ---------------------------------------------------------------------------
#  Данные
# ---------------------------------------------------------------------------

def read_dialogs(path: str) -> list[dict]:
    """Прочитать jsonl-файл и сложить диалоги в список.

    От каждой строки остаются только номер диалога и реплики:
    {"dialog_id": ..., "turns": [{"role": ..., "text": ...}, ...]}.
    """
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    """Ответы по задаче task ("reaction" или "topic"): dialog_id -> метка.

    Диалоги, у которых такой метки нет, пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


# ---------------------------------------------------------------------------
#  Обработка текста, как на семинаре 4
# ---------------------------------------------------------------------------

def normalize(text: str) -> list[str]:
    """Нижний регистр, слова из букв, без стоп-слов, основы слов.

    Язык определяется по доле кириллицы: русский или английский.
    """
    words = WORD.findall(text.lower())
    letters = "".join(words)
    language = "russian" if letters and len(CYRILLIC.findall(letters)) > len(letters) / 2 else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    stems = [stemmer.stem(word) for word in words if len(word) >= 2 and word not in stop]
    return stems


def dialog_text(dialog: dict) -> str:
    """Все реплики пользователя одной строкой из нормализованных слов."""
    user_text = " ".join(turn["text"] for turn in dialog["turns"] if turn["role"] == "user")
    text = " ".join(normalize(user_text))
    return text


# ---------------------------------------------------------------------------
#  Модели: реакция пользователя
# ---------------------------------------------------------------------------

def init_reaction_models(train_path: str) -> dict[str, DummyClassifier]:
    """Все модели задачи 1: имя -> готовая к предсказанию модель.

    В примере модели обучаются прямо здесь на размеченных диалогах
    из train_path. В вашем решении здесь загрузка из artifacts.
    """
    texts, labels = _training_data(train_path, "reaction")
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent").fit(texts, labels),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=0).fit(texts, labels),
    }
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: "positive", "neutral" или "negative"."""
    texts = [dialog_text(dialog) for dialog in dialogs]
    answers = [str(label) for label in model.predict(texts)]
    return answers


# ---------------------------------------------------------------------------
#  Модели: темы
# ---------------------------------------------------------------------------

def init_topic_models(train_path: str) -> dict[str, DummyClassifier]:
    """Все модели задачи 2: имя -> готовая к предсказанию модель.

    Меток тем в данных нет, темы придумываете вы. В примере вместо ваших тем
    две заглушки: одна тема на все диалоги и десять случайных тем.
    """
    dialog_ids = [dialog["dialog_id"] for dialog in read_dialogs(train_path)]
    one_topic = ["все диалоги"] * len(dialog_ids)
    ten_topics = [f"тема {k % 10}" for k in range(len(dialog_ids))]
    models = {
        "dummy_one_topic": DummyClassifier(strategy="most_frequent").fit(dialog_ids, one_topic),
        "dummy_random_topics": DummyClassifier(strategy="uniform", random_state=0).fit(dialog_ids, ten_topics),
    }
    return models


def predict_topic(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: название темы."""
    texts = [dialog_text(dialog) for dialog in dialogs]
    answers = [str(label) for label in model.predict(texts)]
    return answers


def _training_data(train_path: str, task: str) -> tuple[list[str], list[str]]:
    labels = read_labels(train_path, task)
    dialogs = [dialog for dialog in read_dialogs(train_path) if dialog["dialog_id"] in labels]
    texts = [dialog_text(dialog) for dialog in dialogs]
    targets = [labels[dialog["dialog_id"]] for dialog in dialogs]
    return texts, targets


# ---------------------------------------------------------------------------
#  Метрики и сохранение ответов
# ---------------------------------------------------------------------------

def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """Метрики по тем диалогам, для которых известен ответ."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    return metrics


def compute_partition_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """Сравнение разбиений для тем: названия тем не обязаны совпадать,
    сравнивается, какие диалоги попали в одну тему."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    metrics = {"ARI": adjusted_rand_score(y_true, y_pred)}
    return metrics


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])