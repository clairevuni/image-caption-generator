import os
import csv
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_clean_captions_csv(path):
    """
    Carica caption da un file CSV con intestazione: image,caption
    Ritorna un dizionario: {image_id: [caption list]}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    captions = {}

    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_filename = row['image']
            caption = row['caption'].lower().strip().replace('.', '')
            img_id = img_filename.split('.')[0]
            full_caption = f"startseq {caption} endseq"
            captions.setdefault(img_id, []).append(full_caption)

    return captions


def load_captions_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    captions = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) < 1:
                continue
            if line.lower().startswith('image'):
                print("Skipping header:", line)  # Debug: verifica la riga header
                continue
            img_file, caption = line.split(',', 1)
            img_id = img_file.split('.')[0]
            caption = caption.lower().strip().rstrip('.')
            full_caption = f"startseq {caption} endseq"
            captions.setdefault(img_id, []).append(full_caption)
    return captions


def to_caption_list(captions_dict):
    """Converte il dizionario delle caption in una lista flat di frasi."""
    all_captions = []
    for cap_list in captions_dict.values():
        all_captions.extend(cap_list)
    return all_captions

def create_tokenizer(captions):
    """Crea un tokenizer e lo adatta alle caption."""
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def max_caption_length(captions):
    """Restituisce la lunghezza massima tra tutte le caption."""
    return max(len(c.split()) for c in captions)

def data_generator(captions_dict, image_features, tokenizer, max_length, vocab_size, batch_size):
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    import random

    while True:
        X1, X2, y = [], [], []

        while len(X1) < batch_size:
            img_id = random.choice(list(captions_dict.keys()))
            caption_list = captions_dict[img_id]
            caption = random.choice(caption_list)

            seq = tokenizer.texts_to_sequences([caption])[0]

            if len(seq) < 2:
                continue  # Skip captions troppo corte

            # Prendi un token casuale per predire
            i = random.randint(1, len(seq)-1)
            in_seq = seq[:i]
            out_seq = seq[i]

            in_seq_padded = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
            out_seq_cat = to_categorical(out_seq, num_classes=vocab_size)

            # --- LA CORREZIONE È QUI ---
            # Usiamo np.squeeze per assicurare che la forma sia (2048,) e non (1, 2048)
            feature_vector = np.squeeze(image_features[img_id])
            X1.append(feature_vector.astype(np.float32))
            # ---------------------------
            
            X2.append(in_seq_padded.astype(np.int32))
            y.append(out_seq_cat.astype(np.float32))

        # Converto tutto in numpy array
        batch_X1 = np.array(X1)
        batch_X2 = np.array(X2)
        batch_y = np.array(y)
        
        # Le stampe di debug sono ottime per verificare, lasciale pure per ora
        # print(f"Batch shapes: X1={batch_X1.shape}, X2={batch_X2.shape}, y={batch_y.shape}")
        # print(f"Batch dtypes: X1={batch_X1.dtype}, X2={batch_X2.dtype}, y={batch_y.dtype}")

        yield (batch_X1, batch_X2), batch_y