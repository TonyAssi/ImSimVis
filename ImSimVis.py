import csv
import os
import requests
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModel
from huggingface_hub import create_repo
import numpy as np
from sklearn.decomposition import PCA
from huggingface_hub import HfApi


def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

def create_ds_app(input_csv, dataset_name, token):
    # Download images
    print('Downloading images...')

    output_folder = 'images'
    metadata_csv = 'images/metadata.csv'

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare metadata file
    with open(metadata_csv, mode='w', newline='') as meta_file:
        meta_writer = csv.writer(meta_file)
        meta_writer.writerow(['file_name', 'id', 'image_url'])  # Metadata header
        
        # Open input CSV file
        with open(input_csv, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            count = 0
            for row in reader:
                image_url = row['image_url']
                file_name = f"{count:05d}.png"  # Generate 00000.png, 00001.png, etc.
                file_path = os.path.join(output_folder, file_name)
                
                try:
                    # Download image
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()  # Raise error for bad status codes
                    with open(file_path, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                    
                    # Write to metadata
                    meta_writer.writerow([file_name, count, image_url])
                    count += 1
                
                except requests.RequestException as e:
                    print(f"Failed to download {image_url}: {e}")

    # Create dataset
    print('Creating dataset...')
    dataset = load_dataset('imagefolder', data_dir='./images',  split='train')

    # Generate embeddings
    print('Generating embeddings...')

    # Load model for computing embeddings of the candidate images
    model_ckpt='google/vit-base-patch16-224'
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    hidden_dim = model.config.hidden_size

    # Extract embeddings
    dataset_with_embeddings = dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"].convert('RGB'), extractor, model)})

    # Convert embeddings to XY
    print('Convert embeddings to XY...')

    # Convert the 'embedding' column to a numpy array if needed
    embeddings = np.array([np.array(embed) for embed in dataset_with_embeddings['embeddings']])

    # Apply PCA for dimensionality reduction (2D)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    final_dataset = dataset_with_embeddings.add_column("x", reduced_embeddings[:, 0].tolist())
    final_dataset = final_dataset.add_column("y", reduced_embeddings[:, 1].tolist())

    # Create repo
    print('Creating repo...')
    repo_url = create_repo(dataset_name, token=token, repo_type="dataset")
    repo_id = "/".join(repo_url.split('/')[-2:])

    # Upload dataset
    print('Upload data...')
    final_dataset.push_to_hub(repo_id, token=token)
    print('Dataset uploaded to:', repo_url)

    # Create space
    print('Creating space...')
    space_url = create_repo(dataset_name + '-app', token=token, repo_type="space", space_sdk='streamlit')
    space_id = "/".join(space_url.split('/')[-2:])

    print('App created:', space_url )
    print('Space ID:',  space_id)

    # Change file content
    # app.py
    print('Generating app files...')
    with open('app/app.py', 'r') as file:
        file_contents = file.read()

    # Replace "DATASET_ID" with "hey"
    file_contents = file_contents.replace("DATASET_ID","'" + repo_id + "'" )
    file_contents = file_contents.replace("DATASET_URL",repo_url)

    # Open the file in write mode and overwrite with modified content
    with open('app/app.py', 'w') as file:
        file.write(file_contents)

    # README.md
    with open('app/README.md', 'r') as file:
        file_contents = file.read()

    # Replace "DATASET_ID" with "hey"
    file_contents = file_contents.replace("SPACE_TITLE", space_id.split('/')[1])

    # Open the file in write mode and overwrite with modified content
    with open('app/README.md', 'w') as file:
        file.write(file_contents)

    # Upload files to space
    print('Uploading app files...')
    api = HfApi()
    api.upload_folder(
        folder_path="./app",
        repo_id=space_id,
        repo_type="space",
        token=token
    )

    print("""
********* SUCCESS *******************
    
SUMMARY
Dataset URL: """ + repo_url + """
App URL: """ + space_url + """

    """)
