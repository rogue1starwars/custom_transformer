import spacy
import os


def load_tokenizers():
    """Load the Japanese and English tokenizers using spaCy.

    This function attempts to load the tokenizers for Japanese and English.
    If the models are not found, it will download them automatically.

    Returns:
        tuple: A tuple containing the Japanese and English tokenizers.
    """

    try:
        spacy_ja = spacy.load("ja_core_news_sm")
    except IOError:
        os.system("python -m spacy download ja_core_news_sm")
        spacy_ja = spacy.load("ja_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_ja, spacy_en
