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

def data_generator_single_example(captions_dict, image_features, tokenizer, max_length, vocab_size):
    """
    Generatore che restituisce UN solo esempio alla volta.
    """
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    import random

    # Il ciclo principale ora genera un esempio e lo restituisce subito
    while True:
        # Scegli un'immagine casuale
        img_id = random.choice(list(captions_dict.keys()))
        
        # Controlliamo che l'immagine esista nel dizionario delle feature
        if img_id not in image_features:
            continue
            
        # Scegli una caption casuale per quell'immagine
        caption = random.choice(captions_dict[img_id])
        
        # Tokenizza la caption
        seq = tokenizer.texts_to_sequences([caption])[0]

        # Creiamo coppie input/output dalla sequenza
        # Iteriamo su ogni parola della caption per creare più esempi
        for i in range(1, len(seq)):
            # La sequenza di input è fino alla parola i-esima
            in_seq = seq[:i]
            # La sequenza di output è la parola i-esima
            out_word = seq[i]
            
            # Esegui il padding della sequenza di input
            in_seq_padded = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
            
            # Converti la parola di output in one-hot encoding
            out_word_one_hot = to_categorical([out_word], num_classes=vocab_size)[0]
            
            # Estrai le feature dell'immagine
            img_feature = image_features[img_id]

            # Fai yield di un singolo campione completo
            yield (img_feature, in_seq_padded), out_word_one_hot

# Puoi anche lasciare la vecchia funzione data_generator nel file se vuoi, non darà fastidio
# a meno che non la chiami.