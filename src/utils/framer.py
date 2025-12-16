import os
import numpy as np
from pathlib import Path

from framer_functions import read_images, save_images, dataset_check, process_mediapipe, extract_frames

import logging
logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# USER SETTINGS.
SOURCE_FOLDER = './datasets/Bukva/'

# Каталог с исходниками (in SOURCE_FOLDER).
# 'frames' - кадры уже нарезаны.
# 'video' - необходимо из видео нарезать кадры.
SOURCE_TYPE = 'video'
# Каталог, в который сохранятся кадры (in SOURCE_FOLDER).
OUT_FOLDER = 'framer_output'
# Перечень файлов (in SOURCE_FOLDER).
ANNOTATIONS_FILE_NAME = 'annotations.csv'

# Сколько кадров нужно извлечь.
N_FRAMES = 40

READ_RESIZE_FLAG = True
SAVE_RESIZE_FLAG = True

# Наносить метки на кадры.
MP_FLAG = True

# Метод отбора кадров:
# 'window'  - делит видео на сегменты и ищет САМЫЙ РЕЗКИЙ внутри сегмента (лучше качество).
# 'uniform' - берет кадры строго равномерно по времени (лучше для таймингов, но кадры могут быть смазаны).
SELECTION_METHOD = 'window'

# Корневые папки.
INPUT_DIR_PATH = Path(SOURCE_FOLDER) / SOURCE_TYPE
OUTPUT_DIR_PATH = Path(SOURCE_FOLDER) / OUT_FOLDER

if not os.path.exists(OUTPUT_DIR_PATH):
    os.makedirs(OUTPUT_DIR_PATH)

csv_path = Path(SOURCE_FOLDER) / ANNOTATIONS_FILE_NAME

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Annotations has not been found: {csv_path}")

annotations = np.genfromtxt(
    csv_path,
    delimiter=',',
    usecols=0, # 'attachment_id'
    skip_header=1,
    dtype=str,          # ← critical: read as string
    encoding='utf-8'    # ← good practice for non-English text
)

logger.info(f"START: method {SELECTION_METHOD}, target: {N_FRAMES} frames in {len(annotations)} items.")

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')


# Основной алгоритм.
if SOURCE_TYPE == 'frames':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isdir(os.path.join(INPUT_DIR_PATH, item))
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    # Check weather dir contains all items from annotation list.
    dataset_check(items, annotations)
    
    for item in items:          
        input_item_dir_path = INPUT_DIR_PATH / item
        output_item_dir_path = OUTPUT_DIR_PATH / item

        # Load images from folder.
        img_set = read_images(input_item_dir_path, resize_flag=READ_RESIZE_FLAG)
        
        if len(img_set) != N_FRAMES:
            raise ValueError(f"{len(img_set)} of {N_FRAMES} image files found in '{INPUT_DIR_PATH / item}'.")
        
        if MP_FLAG:
            img_set, annotated_cnt = process_mediapipe(img_set)
            logger.info(f"Кисть найдена на {annotated_cnt} из {len(img_set)} кадров ({item}).")

        save_images(img_set, output_item_dir_path, resize_flag=SAVE_RESIZE_FLAG)
    
elif SOURCE_TYPE == 'video':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isfile(os.path.join(INPUT_DIR_PATH, item)) 
        and item.lower().endswith(video_extensions)
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    # Check weather dir contains all items from annotation list.
    dataset_check([os.path.splitext(item)[0] for item in items], annotations)
    
    for item in items:  
        item_name, _ = os.path.splitext(item)
        input_item_dir_path = INPUT_DIR_PATH / item
        output_item_dir_path = OUTPUT_DIR_PATH / item_name
    
        # Cut video into frames.
        img_set = extract_frames(
            video_path=input_item_dir_path,
            num_frames=N_FRAMES,
            method=SELECTION_METHOD,
            resize_flag=READ_RESIZE_FLAG
        )

        if len(img_set) != N_FRAMES:
            raise ValueError(f"{len(img_set)} of {N_FRAMES} image files found in '{INPUT_DIR_PATH / item}'.")
        
        if MP_FLAG:
            img_set, annotated_cnt = process_mediapipe(img_set)
            logger.info(f"Размечены {annotated_cnt} из {len(img_set)} кадров ({item}).")

        save_images(img_set, output_item_dir_path, resize_flag=SAVE_RESIZE_FLAG)
else:
    raise ValueError(f"Wrong 'SOURCE_TYPE'.")