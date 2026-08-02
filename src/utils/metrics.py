import re
import jiwer

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    # Remove standard punctuation using a specific set, preserving all word characters including Vietnamese
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_wer(references: list, predictions: list) -> float:
    if not references or not predictions:
        return None
    return jiwer.wer(references, predictions) * 100
