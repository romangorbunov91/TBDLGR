import os
import numpy as np
from pathlib import Path

import utils.frame_functions as frame_functions
from frame_functions import process_mediapipe

import logging
logger = logging.getLogger(__name__)

# USER SETTINGS.
SOURCE_FOLDER = './datasets/Bukva/'

# Каталог с исходниками (in SOURCE_FOLDER).
# 'frames' - кадры уже нарезаны.
# 'video' - необходимо из видео нарезать кадры.
SOURCE_TYPE = 'frames'
# Каталог, в который сохранятся кадры (in SOURCE_FOLDER).
OUT_FOLDER = 'framer_output'
# Перечень файлов (in SOURCE_FOLDER).
ANNOTATIONS_FILE_NAME = 'annotations.csv'

# Сколько кадров нужно извлечь
N_FRAMES = 40

# Resize outputs to 224x224.
RESIZE_FLAG = True

# Наносить метки на кадры.
MP_FLAG = True

# Метод отбора кадров:
# 'window'  - делит видео на сегменты и ищет САМЫЙ РЕЗКИЙ внутри сегмента (лучше качество)
# 'uniform' - берет кадры строго равномерно по времени (лучше для таймингов, но кадры могут быть смазаны)
SELECTION_METHOD = 'window'

# Корневые папки.
INPUT_DIR_PATH = Path(SOURCE_FOLDER + SOURCE_TYPE)
OUTPUT_DIR_PATH = Path(SOURCE_FOLDER + OUT_FOLDER)

if not os.path.exists(OUTPUT_DIR_PATH):
    os.makedirs(OUTPUT_DIR_PATH)

csv_path = SOURCE_FOLDER + ANNOTATIONS_FILE_NAME

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

logger.info(f"Start: method: {SELECTION_METHOD}, target: {N_FRAMES} frames in {len(annotations)} items.")

for root, dirs, files in os.walk(INPUT_DIR_PATH):
    logger.info(f"{len(files)} items found in {root}.")

    match SOURCE_TYPE:
        case 'frames':
            output_item_dir_path = root.replace(SOURCE_TYPE, OUT_FOLDER)
            if not os.path.exists():
                os.makedirs(output_item_dir_path, exist_ok=True)
        #case 'video':
            
    files_found = []

    for i, file in enumerate(files):
        
        f_name, _ = os.path.splitext(file)
        
        if f_name in annotations:
        
            # Пропускаем не видео файлы (опционально).
            if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue

            full_video_path = os.path.join(root, file)
            
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
            
            logger.info("-" * N_FRAMES)
            files_found.append(f_name)

    if len(files_found) != len(annotations):
        annot_left = [item for item in annotations.tolist() if item not in set(files_found)]
        raise FileNotFoundError(f"Files from {csv_path} has not been found: {annot_left}")



