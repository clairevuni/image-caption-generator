import os
import glob
import numpy as np
import spacy
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# Carica modello spaCy e aumenta la lunghezza massima
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4_000_000

def load_and_preview_images(images_path, preview_count=9):
    """
    Carica e visualizza le prime immagini da una directory.
    """
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Directory not found: {images_path}")

    image_files = glob.glob(os.path.join(images_path, "*.jpg"))
    if len(image_files) == 0:
        raise FileNotFoundError("No images found in the directory.")

    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(image_files[:preview_count]):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_array)
        plt.axis('off')
    plt.show()


def preprocess_with_spacy(text_path):
    """
    Carica un file di testo e restituisce una lista di lemmi filtrati.
    """
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    doc = nlp(text.lower())

    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    return tokens


def clean_text(path):
    """
    Pulizia completa del testo da file: lettura, preprocessing, e join finale.
    """
    if os.path.exists(path):
        tokens = preprocess_with_spacy(path)
        return ' '.join(tokens)
    else:
        raise FileNotFoundError("Text file path not found.")
