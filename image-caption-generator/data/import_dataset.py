import kagglehub
import os
import shutil

# Download the dataset from Kaggle!
source_path = kagglehub.dataset_download("adityajn105/flickr8k")

# The file will be downloaded to the source_path
print("Dataset scaricato in:", source_path)


# then, we want the images and the text, in another folder!
destination_path = "data/Flickr8k_Dataset"
os.makedirs(destination_path, exist_ok=True)

# So for everything in the source_path (will be a cache folder), we have a source and destination path! 
# we copy the files to the destination_path using shutil and copytree. 
for item in os.listdir(source_path):
    s = os.path.join(source_path, item)
    d = os.path.join(destination_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)


#then we print the destination path
print("File copiati in:", destination_path)
