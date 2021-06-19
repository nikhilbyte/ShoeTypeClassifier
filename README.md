# ShoeTypeClassifier
A simple Visual Transformer based Shoe Type Classifier.


## Project Structure
The descriptions of principal files in this project are as follows:
- `main.py`: codes for using the Model to check the categories of images inside a folder.
- `ViT_Fine_Tuning.ipynb`: codes for model creation and fine tuning on our dataset.
- `Dataset_Creation.ipynb`: codes for dataset creationg using the bing image downloader.


## Formats and descriptions of the dataset:
data format: records of image and their correspondig labels

| "image_id" | "label" |
|  ----  | ----  |
| image | label |


## How to use
please generate the models' assets by fine tuning on the dataset of your choice using the `ViT_Fine_Tuning.ipynb` and modify the path in `main.py`.

Principal environmental dependencies as follows:
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [sklearn](https://scikit-learn.org/stable/)
- [vit-keras](https://github.com/faustomorales/vit-keras.git)
- [tensorflow==2.4.1](https://www.tensorflow.org/)
