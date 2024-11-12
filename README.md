# Image Similarity Data Visualization
by [Tony Assi](https://www.tonyassi.com/)

Create a 2d scatter plot data visualization of image similarity. All you need is a .csv file of image urls.

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
The script will download images, generate image embeddings, upload the dataset to Hugging Face, and create a visualization app.
