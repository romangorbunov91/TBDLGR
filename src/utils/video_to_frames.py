import os
import logging
import numpy as np
from pathlib import Path

# Импортируем наш модуль
import sharper_frames as sharper_frames

# ================= НАСТРОЙКИ ЛОГИРОВАНИЯ =================
# Настраиваем один раз в точке входа
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ================= ГИПЕРПАРАМЕТРЫ =================
# Перечень файлов.
annotations_file_name = 'annotations.csv'

# Папка, где лежат исходные видео
SOURCE_FOLDER = './datasets/Bukva/'

# Корневые папки для результатов
OUTPUT_SHARP_ROOT = Path('./datasets/Bukva/frames')          # Чистые кадры
#OUTPUT_MPIPE_ROOT = Path('sharp_frames_mpipe')    # Кадры с разметкой

# Сколько кадров нужно извлечь
TARGET_FRAMES = 40

# Метод отбора кадров:
# 'window'  - делит видео на сегменты и ищет САМЫЙ РЕЗКИЙ внутри сегмента (лучше качество)
# 'uniform' - берет кадры строго равномерно по времени (лучше для таймингов, но кадры могут быть смазаны)
SELECTION_METHOD = 'window' 

# ================= ОСНОВНОЙ ЦИКЛ =================
if __name__ == "__main__":

    csv_path = SOURCE_FOLDER + annotations_file_name
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations has not been found: {csv_path}")

    annotations = np.genfromtxt(
        csv_path,
        delimiter=',',
        usecols=1,
        skip_header=1,
        dtype=str,          # ← critical: read as string
        encoding='utf-8'    # ← good practice for non-English text
    )

    if not os.path.exists(SOURCE_FOLDER + 'video'):
        logger.critical(f"Исходная папка не найдена: {SOURCE_FOLDER + 'video'}")
        exit(1)

    logger.info(f"Начинаем обработку. Метод: {SELECTION_METHOD}, Цель: {TARGET_FRAMES} кадров.")

    # Рекурсивный обход папок
    for root, dirs, files in os.walk(SOURCE_FOLDER + 'video'):
        logger.info(f"Сканирование папки: {root}. Найдено файлов: {len(files)}")
        
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
                
                # Формируем пути для конкретного видео
                # sharp_frames/video1/
                current_sharp_output = OUTPUT_SHARP_ROOT / video_stem
                # sharp_frames_mpipe/video1/
                #current_mpipe_output = OUTPUT_MPIPE_ROOT / video_stem
                
                # 1. Запускаем извлечение кадров (через модуль)
                saved_frames = sharper_frames.extract_frames(
                    video_path=full_video_path, 
                    output_folder=current_sharp_output,
                    num_frames=TARGET_FRAMES,
                    method=SELECTION_METHOD
                )
                '''
                # 2. Запускаем валидацию MediaPipe (если кадры извлеклись)
                if saved_frames:
                    sharper_frames.process_mediapipe(
                        pictures=saved_frames, 
                        out_dir_name=current_mpipe_output
                    )
                '''
                logger.info("-" * TARGET_FRAMES)
                files_found.append(f_name)

        if len(files_found) != len(annotations):
            annot_left = [item for item in annotations.tolist() if item not in set(files_found)]
            raise FileNotFoundError(f"Files from {csv_path} has not been found: {annot_left}")