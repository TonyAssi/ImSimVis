# Image Search with Text Prompt
by [Tony Assi](https://www.tonyassi.com/)

Use text prompt to search for images in folders. It can search through folders within folders. Built with 🤗 Transformers and 🤗 Datasets.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Import the module
```python
from ImageSearch import ImageSearch
```
Define folder path
- **image_dir** Parent folder
- **hf_key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
image_search = ImageSearch(image_dir='images',
			   hf_key='HF_KEY')
```
The first time this is called the images will be converted to a 🤗 Dataset and it'll get uploaded to the 🤗 Hub. It'll print out the dataset id and store it in a meta.text file. It should look like [tonyassi/images-ds](https://huggingface.co/datasets/tonyassi/images-ds). It'll take a little longer the first time it's called. After the dataset is created it should be very quick.



Search for image with text prompt
- **text** Text prompt
- **download_path** Images most similar to text prompt will be downloaded to this path (if download_path='' then images will not be downloaded)
- **num** Number of images (optional) defaults to 5
```python
image_search.search(text='red rose',
		    download_path = 'found_images',
		    num=5)
```
The most similar images will be printed and downloaded.

![0](https://github.com/user-attachments/assets/74f515c2-fec2-491a-96ba-be8d3b9427de)
![1](https://github.com/user-attachments/assets/a5bcfa3f-fa75-455b-970d-38f7f28259c4)
![2](https://github.com/user-attachments/assets/ccec1e8a-c750-4442-80d5-9c509b4470c2)

 ```
flowers/0092.png
Score: 142.48643
Index: 0

flowers/0105.png
Score: 146.4621
Index: 1

flowers/0095.png
Score: 148.10144
Index: 2
```
