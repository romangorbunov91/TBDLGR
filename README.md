# Transformer-Based Dactile Language Gesture Recognition

Решена задача перевода (распознавания, классификации) изолированных жестов русского дактильного языка. Область применения - тренажеры алфавита языка жестов.

<img src='img/gesture_set.png' style='width:100%; height:auto;'>

## Архитектура модели

В основу проекта положена архитектура, описанная в статье

*A. D’Eusanio, A. Simoni, S. Pini, G. Borghi, R. Vezzani, R. Cucchiara*  
**A Transformer-Based Network for Dynamic Hand Gesture Recognition**  
*In International Conference on 3D Vision (3DV) 2020.*

**[[Paper](https://iris.unimore.it/retrieve/handle/11380/1212263/282584/3DV_2020.pdf)]  [[Project Page](https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=32)]**

<p align="center" width="100%">
  <img src="./img/model.png"
  style="background-color: white; padding: 0;
  width="100%" />
</p>

В статье предложена архитектура для задачи распознавания динамических жестов в системах автоматики. В том числе со сбором информации одновременно с разных типов датчиков.
<!-- 
<p align="center" width="100%">
  <img src="./img/briareo.gif" width="70%" />
</p>
-->

## Подготовка датасета
Архитектура настроена на работу с кадрами в качестве входных данных.

Преобразование видео-данных в наборы кадров выполнено скриптом [framer.py](src\utils\framer.py). Скрипт реализует следующую последовательность операций.

## Обучение


Веса в каталоге

Инструкция по интеграции весов (правки в )

## Использование модели

### 1. Создайте окружение в директории `.venv`
```
python -m venv .venv
```
### 2. Активируйте окружение
```
.venv\Scripts\activate
```
### 3. Установите библиотеки
```
pip install -r requirements.txt
```
### 4. В файле `\.venv\Lib\site-packages\imgaug\imgaug.py` замените сроки
```
NP_FLOAT_TYPES = ...
NP_INT_TYPES = ...
NP_UINT_TYPES = ...
```
на следующие:
```
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
```
### 5. Скачайте веса предобученной модели
[Веса модели](?? "Pretrained weights") поместите в директорию `checkpoints/Bukva/`.

### 6. Запустите DEMO (Streamlit)

...

## Работа с архитектурой
### Файл конфигурации и структура проекта
Гиперпараметры задаются в файле [config.json](src\hyperparameters\Bukva\train.json "hyperparameters").

Основные параметры:
- **phase** - "train" или "test".
- **data-->data_path** - путь к каталогу `frames`, содержащему датасет в кадрах.
- **data-->n_classes** - количество классов в датасете.

Рекомендуемая структура проекта:

```
project/
├── datasets/
│   └── Bukva/
│       └── frames/
│           └── f4356671bbe8c7e3c0a3c9c54e5b713e/
│           └── fd77732d-9188-4baa-a678-b5fb7298c13f/
│           └── ...
│       └── annotations.csv
├── checkpoints/
│   └── Bukva/
│       └── *.pth
...
├── README.md
└── requirements.txt
```

### Датасет
Датасет необходимо разместить в директории `datasets/Bukva/`.

Структура датасета:
- `frames` - каталог, содержащий набор подкаталогов по `n_frames`-картинок в каждом.
- `annotations.csv` - файл аннотаций.

### Веса
Реализована возможность подгрузить веса модели - как на `test`, так и на дообучение в `train`. Файл весов `*.pth` необходимо разместить в директории `checkpoints/Bukva/`.

### Запуск на обучение
```
python src/main.py --hypes src/hyperparameters/Bukva/config.json 
```

### Запуск на дообучение 
```
python src/main.py --hypes src/hyperparameters/Bukva/config.json --resume checkpoints/Bukva/best_train_bukva.pth
```

### Запуск на тест 
```
python src/main.py --hypes src/hyperparameters/Bukva/config.json --resume checkpoints/Bukva/best_train_bukva.pth --phase test
```

## Авторы

* [Роман Горбунов](https://github.com/romangorbunov91)
* [Станислава Иваненко](https://github.com/smthCreate)
* [Максим Шугаев](https://github.com/knjii)
* [Анжелина Абдулаева](https://github.com/anzhelina0)
* [Кирилл Зайцев]()


## About [Bukva&copy;](https://github.com/ai-forever/bukva?tab=readme-ov-file#bukva-russian-sign-language-alphabet-dataset)

<figure>
  <img src='img/bukva_pipeline.png'
      style='width:100%; height:auto;'>
  <figcaption>↑Последовательность создания датасета Bukva.</figcaption>
</figure>

Запись видеороликов осуществлялась разными людьми по инструкции разработчиков датасета.


<figure>
  <img src='img/bukva.gif'
      style='width:100%; height:auto;'>
  <figcaption>↑Эталонная демонстрация жестов (инструкция).</figcaption>
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
  <figcaption>↑Фрагменты итогового датасета.</figcaption>
</figure>

На последнем этапе размечены интервалы. Разметка осуществлялась тремя разными пользователями с последующей агрегацией интервалов.

<figure>
  <img src='img/trimmer_marks.png'
      style='width:100%; height:auto;'>
  <figcaption>↑Разметка интервалов видеороликов.</figcaption>
</figure>

**Подробнее**
- [Bukva: алфавит русского жестового языка](https://habr.com/ru/companies/sberdevices/articles/850858/)
- [Bukva: Russian Sign Language Alphabet](https://arxiv.org/abs/2410.08675)

**Скачать датасет Bukva**

- **[Bukva-video Full Official](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip "Bukva: Russian Sign Language Alphabet Dataset")**
- **[Bukva-video Trimmed only](https://drive.google.com/drive/folders/1rXMtY4ja6oxHKdgiV-5taWaEJc1R3kjN?usp=sharing "G-Drive copy")**