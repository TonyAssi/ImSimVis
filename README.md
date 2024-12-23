# Image Similarity Data Visualization
by [Tony Assi](https://www.tonyassi.com/)

Create a 2d scatter plot data visualization of image similarity. All you need is a .csv file of image urls.
<img width="1487" alt="Screenshot 2024-11-12 at 3 12 44 PM" src="https://github.com/user-attachments/assets/dc56a7f0-57f5-4075-a895-9f3dc946d7f0">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/tonyassi/images-data-vis-app)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Import the module
```python
from ImSimVis import create_ds_app
```

- **input_csv** the .csv file containing your list of image urls **(the column must be called 'image_url')**
- **data_name** the name of the Hugging Face repo that the data and app will be uploaded to
- **token** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
create_ds_app(input_csv='image_urls.csv',
	      dataset_name='images-data-vis',
	      token='YOUR_HF_TOKEN')
```
The script will download images, generate image embeddings, upload the dataset to Hugging Face, and create a visualization app. It'll print out the URL to the dataset and app.

[Example Dataset](https://huggingface.co/datasets/tonyassi/images-data-vis)

[Example App](https://huggingface.co/spaces/tonyassi/images-data-vis-app)
