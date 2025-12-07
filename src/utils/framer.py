import os
import numpy as np
from pathlib import Path

from frame_functions import dataset_check, read_images, process_mediapipe

import logging
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

# Resize outputs to 224x224.
# Prior transforms.
PRIOR_RESIZE_FLAG = False
# After all transforms.
AFTER_RESIZE_FLAG = True

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
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

if SOURCE_TYPE == 'frames':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isdir(os.path.join(INPUT_DIR_PATH, item))
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    #dataset_check(items, annotations)
    
    for item in items:          
        output_item_dir_path = OUTPUT_DIR_PATH / item
        if not os.path.exists(output_item_dir_path):
            os.makedirs(output_item_dir_path, exist_ok=True)
    
    img_set = read_images(INPUT_DIR_PATH / item, image_extensions)
    
    if len(img_set) != N_FRAMES:
        raise ValueError(f"{len(img_set)} of {N_FRAMES} image files found in '{INPUT_DIR_PATH / item}'.")
    
    
elif SOURCE_TYPE == 'video':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isfile(os.path.join(INPUT_DIR_PATH, item)) 
        and item.lower().endswith(video_extensions)
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    #dataset_check(items, annotations)
    
    for item in items:  
        item_name, _ = os.path.splitext(item)
        output_item_dir_path = OUTPUT_DIR_PATH / item_name
        if not os.path.exists(output_item_dir_path):
            os.makedirs(output_item_dir_path, exist_ok=True)
            
else:
    raise ValueError(f"Wrong 'SOURCE_TYPE'.")

        




'''
# Имя видео без расширения для названия папки (например "video1")
            video_stem = os.path.splitext(file)[0]
            
            current_sharp_output = OUTPUT_SHARP_ROOT / video_stem
            current_mpipe_output = OUTPUT_MPIPE_ROOT / video_stem
            
            # 1. Запускаем извлечение кадров (через модуль)
            saved_frames = frame_functions.extract_frames(
                video_path=full_video_path, 
                output_folder=current_sharp_output,
                num_frames=N_FRAMES,
                method=SELECTION_METHOD
            )
        
            # 2. Запускаем валидацию MediaPipe (если кадры извлеклись)
            if saved_frames:
                frame_functions.process_mediapipe(
                    pictures=saved_frames, 
                    out_dir_name=current_mpipe_output
                )
            
'''