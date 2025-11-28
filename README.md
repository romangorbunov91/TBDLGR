# A Transformer-Based Dactile Language Gesture Recognition

This is the official PyTorch implementation of the publication:

*A. D’Eusanio, A. Simoni, S. Pini, G. Borghi, R. Vezzani, R. Cucchiara*  
**A Transformer-Based Network for Dynamic Hand Gesture Recognition**  
*In International Conference on 3D Vision (3DV) 2020*

**[[Paper](https://iris.unimore.it/retrieve/handle/11380/1212263/282584/3DV_2020.pdf)]  [[Project Page](https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=32)]**

Transformer-based neural networks represent a successful self-attention mechanism that achieves outstanding results in 
different topics, such as language understanding and sequence modeling. 
The application of such models to different types of data, like the visual one, is necessary to push the boundaries of 
common convolutinal and recurrent neural networks.  
Therefore, in this work we propose a transformer-based architecture for the dynamic hand gesture recognition task, 
focusing on the automotive environment.
Moreover, we propose the combined use of depth maps and surface normals as unique sources to successfully solve the 
task, even in low-light conditions.

<p align="center" width="100%">
  <img src="./img/model.png" width="90%" />
</p>
<p align="center" width="100%">
  <img src="./img/briareo2.gif" width="70%" />
</p>

The two datasets we used are NVGestures and Briareo. Both of them contain data from multiple sensors: RGB, IR, and 
depth, allowing the study of multimodal fusion techniques.

In this work, we focused on the sole use of the depth sensor, which provides light-invariant depth maps that can be 
further processed to obtain an estimation of the surface normals.  
Experimental results show that the use of such a simple processing step leads to a significant gain in accuracy.

<p align="center" width="100%">
  <img src="./img/example_2.png" width="50%" />
</p>


## Getting Started
These instructions will give you a copy of the project up and running on your local machine for development and testing 
purposes. There isn't much to do, just install the prerequisites and download all the files.

### Prerequisites
Create an environment into the folder `.venv`
```
python -m venv .venv
```

Activate the environment
```
.venv\Scripts\activate
```

Run the command:
```
pip install -r requirements.txt
```

In `\.venv\Lib\site-packages\imgaug\imgaug.py` replace to
```
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
```

## Download datasets
The employed datasets are publicy available: 
- **[NVGestures](https://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional "NVIDIA Dynamic Hand Gesture Dataset")**
- **[Briareo](https://drive.google.com/drive/folders/1OqVd9QheO0lYgLAxJ4AQvDO3nkfFNKNM "Briareo Dataset")**

Once downloaded, unzip anywhere in your drive.

## Pretrained model
Pytorch pretrained models are available at this [link](https://drive.google.com/drive/folders/1VXRmAVNP6dgomkovNu2uaFqwwbLs_GE-?usp=sharing "Pretrained weights").

## Setup configuration
For this project we used a json file, located in the hyperparameters folder, such as:  
`hyperparameters/Briareo/[train.json](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=31 "Briareo")`

In there, you can set several parameters, like:

- **Dataset**, Briareo or NVGestures.
- **phase**, select if training or testing.
- **Data-type**, select which source is used: depth, rgb, ir, surface normals or optical-flow.
- **Data-Nframe**, length of the input sequence, default: 40 frame.
- **Data-path**, path where you downloaded and unzipped the dataset.

For every other information check the file.

## Usage TRAIN
```
python src/main.py --hypes src/hyperparameters/Briareo/train.json 
```
## Usage from saved weights (TEST or continue TRAIN)
```
python src/main.py --hypes src/hyperparameters/Briareo/train.json --resume checkpoints/Briareo/best_train_briareo.pth
python src/main.py --hypes src/hyperparameters/Bukva/train.json --resume checkpoints/Bukva/best_train_bukva.pth
```

## Authors

* [Роман Горбунов](https://github.com/romangorbunov91)
* [Станислава Иваненко](https://github.com/smthCreate)
* [Максим Шугаев](https://github.com/knjii)
* [Анжелина Абдулаева](https://github.com/anzhelina0)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

```
pip freeze > requirements.txt
```

```
pip install git+https://github.com/aleju/imgaug.git@0101108d4fed06bc5056c4a03e2bcb0216dac326
```