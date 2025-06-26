# Image Caption Generator with CNN and LSTM

This project implements an **image captioning model** that generates descriptive captions for images. It combines a **Convolutional Neural Network (CNN)** for visual feature extraction with a **Long Short-Term Memory (LSTM)** network for sequence generation.

The project is based on the **Flickr8k** dataset.

---

## Project Structure

image-caption-generator/
│
├── data/
│ ├── Flickr8k_Dataset/
│ │ └── Images/ # All image files
│ └── Flickr8k_text/
│ ├── captions.txt # Raw captions
│ ├── Flickr_8k.trainImages.txt
│ ├── Flickr_8k.devImages.txt
│ └── Flickr_8k.testImages.txt
│
├── features/
│ └── image_features.pkl # Extracted features from images
│
├── src/
│ ├── preprocess.py # Text preprocessing and tokenization
│ ├── extract_features.py # Feature extraction using CNN
│ ├── data_loader.py # Dataset loading and vocabulary setup
│ └── model.py # Model building and training
│
├── notebook/
│ └── exploration.ipynb # Notebooks for data exploration and testing
│
├── results/
│ └── generated_captions.txt # Sample generated captions
│
└── README.md



---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- spaCy (`en_core_web_sm`)
- InceptionV3 (pretrained on ImageNet)

---

## How It Works

1. **Text Preprocessing**
   - Captions are cleaned, lowercased, and lemmatized using spaCy.
   - Stopwords, punctuation, and numbers are removed.

2. **Feature Extraction**
   - Images are processed through InceptionV3.
   - The output from the final average pooling layer (2048-dimensional) is saved.

3. **Dataset Preparation**
   - Captions are tokenized and converted to sequences.
   - Data is split using the official train/dev/test image lists from the dataset.

4. **Model Architecture**
   - Visual features and text input are combined.
   - An LSTM decoder generates captions word by word.

5. **Caption Generation**
   - The model generates a caption for a given image based on learned patterns.

---

## Example Outputs

| Image | Generated Caption |
|-------|-------------------|
| sample1.jpg | "a man riding a horse on the beach" |
| sample2.jpg | "a group of children playing soccer" |

---

## To Do

- Add attention mechanism
- Evaluate using BLEU or METEOR scores
- Extend to Flickr30k or MS-COCO datasets

---

## How to Run

```bash
# Step 1: Extract image features
python src/extract_features.py

# Step 2: Preprocess captions
python src/preprocess.py

# Step 3: Train the model
python src/model.py