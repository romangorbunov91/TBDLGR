# A Transformer-Based Dactile Language Gesture Recognition

Решена задача перевода (распознавания, классификации) изолированных жестов русского дактильного языка. Область применения - тренажеры алфавита языка жестов.

<img src='img/gesture_set.png' style='width:100%; height:auto;'>

## Датасет [Bukva&copy;](https://github.com/ai-forever/bukva?tab=readme-ov-file#bukva-russian-sign-language-alphabet-dataset)

<figure>
  <img src='img/bukva_pipeline.png'
      style='width:100%; height:auto;'>
  <figcaption>Последовательность создания датасета Bukva.</figcaption>
</figure>

Запись видеороликов осуществлялась разными людьми по инструкции разработчиков датасета.


<figure>
  <img src='img/bukva.gif'
      style='width:100%; height:auto;'>
  <figcaption>Эталонная демонстрация жестов (инструкция).</figcaption>
</figure>

Видеоролики фильтрованы разработчиками датасета по следующим правилам:
- В кадре должен находиться только один человек.
- Не менее 720 пикселей по минимальной стороне.
- Не менее 15 кадров в секунду.
- Отсутствие дубликатов видео.
- Рука, выполняющая жест, должна быть полностью видна в кадре.

В результате получились ~4000 HD видеозаписей демонстрации жеста разными людьми (>100 видеозаписей на каждый жест).

<figure>
  <img src='img/bukva_gestures.png'
      style='width:100%; height:auto;'>
  <figcaption>Фрагменты итогового датасета.</figcaption>
</figure>

На последнем этапе размечены интервалы. Разметка осуществлялась тремя разными пользователями с последующей агрегацией интервалов.

<figure>
  <img src='img/trimmer_marks.png'
      style='width:100%; height:auto;'>
  <figcaption>Разметка интервалов видеороликов.</figcaption>
</figure>

Подробнее:
- [Bukva: алфавит русского жестового языка](https://habr.com/ru/companies/sberdevices/articles/850858/)
- [Bukva: Russian Sign Language Alphabet](https://arxiv.org/abs/2410.08675)

## Архитектура модели

В основу проекта положена архитектура, описанная в статье

*A. D’Eusanio, A. Simoni, S. Pini, G. Borghi, R. Vezzani, R. Cucchiara*  
**A Transformer-Based Network for Dynamic Hand Gesture Recognition**  
*In International Conference on 3D Vision (3DV) 2020.*

**[[Paper](https://iris.unimore.it/retrieve/handle/11380/1212263/282584/3DV_2020.pdf)]  [[Project Page](https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=32)]**

В статье предложена архитектура для задачи распознавания динамических жестов в системах автоматики. В том числе со сбором информации одновременно с разных типов датчиков.

<p align="center" width="100%">
  <img src="./img/model.png" width="90%" />
</p>
<p align="center" width="100%">
  <img src="./img/briareo.gif" width="70%" />
</p>


## Подготовка датасета


## Обучение


Веса в каталоге

Инструкция по интеграции весов (правки в )

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

- **[Bukva-video Full Official](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip "Bukva: Russian Sign Language Alphabet Dataset")**
- **[Bukva-video Trimmed only](https://drive.google.com/drive/folders/1rXMtY4ja6oxHKdgiV-5taWaEJc1R3kjN?usp=sharing "G-Drive copy")**

## Pretrained model
Pytorch pretrained models are available at this [link](?? "Pretrained weights").

## Setup configuration
For this project we used a json file [train.json](src\hyperparameters\Bukva\train.json "hyperparameters"), located in the `hyperparameters` folder.


In there, you can set several parameters, like:

- **Dataset**, Briareo or NVGestures.
- **phase**, select if training or testing.
- **Data-type**, select which source is used: depth, rgb, ir, surface normals or optical-flow.
- **Data-Nframe**, length of the input sequence, default: 40 frame.
- **Data-path**, path where you downloaded and unzipped the dataset.

## Usage TRAIN
```
python src/main.py --hypes src/hyperparameters/Bukva/train.json 
```
## Usage from saved weights (TEST or continue TRAIN)
```
python src/main.py --hypes src/hyperparameters/Briareo/train.json --resume checkpoints/Briareo/best_train_briareo.pth
python src/main.py --hypes src/hyperparameters/Bukva/train.json --resume checkpoints/Bukva/best_train_bukva.pth
```

## Авторы

* [Роман Горбунов](https://github.com/romangorbunov91)
* [Станислава Иваненко](https://github.com/smthCreate)
* [Максим Шугаев](https://github.com/knjii)
* [Анжелина Абдулаева](https://github.com/anzhelina0)
* [Кирилл Зайцев]()

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

```
pip freeze > requirements.txt
```