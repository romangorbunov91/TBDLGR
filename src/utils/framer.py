import os
import numpy as np
import mediapipe as mp
from pathlib import Path
import cv2

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
SOURCE_TYPE = 'frames'
# Каталог, в который сохранятся кадры (in SOURCE_FOLDER).
OUT_FOLDER = 'framer_output'
# Перечень файлов (in SOURCE_FOLDER).
ANNOTATIONS_FILE_NAME = 'annotations.csv'

# Сколько кадров нужно извлечь.
N_FRAMES = 40

# Resize outputs to 224x224.
# Prior transforms.
READ_RESIZE_FLAG = True
# After all transforms.
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
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')


def read_images(dir_path, resize_flag=False):
    img_names = [
        img for img in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, img)) 
        and img.lower().endswith(image_extensions)
        ]
    img_set = []
    for img in img_names:
        if resize_flag:
            img_set.append(cv2.flip(cv2.resize(cv2.imread(os.path.join(dir_path, img)), (224, 224)), 1))
        else:
            img_set.append(cv2.flip(cv2.imread(os.path.join(dir_path, img)), 1))
    
    logger.info(f"Считаны {len(img_set)} кадров {dir_path}")

    return img_set


def save_images(img_set, dir_path, resize_flag=False):
    
    # Create destination folder.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    for idx, img in enumerate(img_set):
        output_filename = os.path.join(dir_path, f"frame_{idx:03d}.jpg")
        if resize_flag:
            success = cv2.imwrite(output_filename, cv2.flip(cv2.resize(img, (224, 224)), 1))
        else:
            success = cv2.imwrite(output_filename, cv2.flip(img, 1))

        if not success:
            raise OSError(f"Ошибка записи: {dir_path}")

    logger.info(f"Сохранены {len(img_set)} кадров {dir_path}")


def dataset_check(dir_items, full_list):
    items_left = [item for item in full_list.tolist() if item not in set(dir_items)]
    if items_left:
        raise FileNotFoundError(f"Files from annotation that has not been found: {items_left}")


def process_mediapipe(img_set):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Инициализация модели MediaPipe Hands.
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        annotated_img_set = []
        # Счетчик количества кадров, на которых распозналась кисть.
        annotated_cnt = 0
        # Перебираем все изображения.
        for img in img_set:

            # Обрабатываем изображение с помощью MediaPipe.
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                annotated_cnt += 1
            annotated_img_set.append(img)
    
    return annotated_img_set, annotated_cnt


def calculate_sharpness(frame):
    """Вычисляет дисперсию Лапласиана (меру резкости)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def get_selected_frame_indices(frame_data, num_frames, method='window'):
    """
    Выбирает индексы кадров на основе метода.
    
    Args:
        frame_data: список кортежей [(frame_index, sharpness_score), ...]
        num_frames: требуемое количество кадров
        method: 'window' (лучший в сегменте) или 'uniform' (равномерный шаг)
        
    Returns:
        Список кортежей, отсортированный по frame_index (хронологически).
    """
    total_frames = len(frame_data)
    
    # Если кадров в видео меньше, чем нужно - берем все и сортируем по времени
    if total_frames < num_frames:
        logger.warning(
            f"В видео всего {total_frames} кадров, запрошено {num_frames}. "
            f"Будет выполнен апсемплинг с дублированием кадров."
        )
        selected_frames = upsample_frame_indices(frame_data, num_frames)
        selected_frames.sort(key=lambda x: x[0])
        return selected_frames

    selected_frames = []

    if method == 'uniform':
        # Равномерное распределение индексов
        # Например, из 100 кадров берем 0, 25, 50, 75, 99
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        selected_frames = [frame_data[i] for i in indices]
        logger.info("Метод 'uniform': кадры взяты с равным шагом.")

    elif method == 'window':
        # Делим видео на сегменты и ищем самый резкий кадр в каждом
        chunk_size = total_frames / num_frames
        
        for i in range(num_frames):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            
            # Корректировка последнего отрезка
            if i == num_frames - 1:
                end = total_frames
            
            # Срез списка кадров для текущего окна
            window_frames = frame_data[start:end]
            
            if window_frames:
                # Находим кадр с максимальной резкостью внутри этого окна
                best_in_window = max(window_frames, key=lambda x: x[1])
                selected_frames.append(best_in_window)
        
        logger.info(f"Метод 'window': выбраны самые резкие кадры из {num_frames} временных сегментов.")

    # ВАЖНО: Сортируем итоговый список по индексу кадра (x[0]), 
    # чтобы сохранить хронологию движения для модели (t0 -> t1 -> tN)
    selected_frames.sort(key=lambda x: x[0])
    
    return selected_frames


def upsample_frame_indices(frame_data, num_frames):
    """
    Апсемплинг кадров, если исходных меньше, чем нужно.
    
    Args:
        frame_data: список кортежей [(frame_index, sharpness_score), ...],
                    ДОЛЖЕН быть отсортирован по frame_index (хронологически).
        num_frames: целевое количество кадров (например, 40)
    
    Returns:
        Список длины num_frames, содержащий (frame_index, sharpness_score),
        с возможными повторами, но в хронологическом порядке.
    """
    total = len(frame_data)
    if total == 0:
        return []
    if total >= num_frames:
        # На всякий случай — апсемплинг тут не нужен.
        return frame_data

    # Гарантируем хронологический порядок исходных кадров
    frame_data = sorted(frame_data, key=lambda x: x[0])

    # Изначально каждый кадр возьмём по одному разу
    repeat_counts = np.ones(total, dtype=int)

    # Сколько дублей надо добавить, чтобы получить num_frames
    extra = num_frames - total  # > 0, так как total < num_frames

    # Равномерно распределим эти дополнительные повторы по временной оси.
    # Идея: линейно интерполируем extra позиций от 0 до total-1.
    extra_indices = np.linspace(0, total - 1, extra, dtype=int)
    # Каждый раз, когда индекс встречается в extra_indices, увеличиваем
    # количество повторов соответствующего кадра на 1
    for idx in extra_indices:
        repeat_counts[idx] += 1

    # Собираем финальный список: кадры идут по времени,
    # каждый дублируется repeat_counts[i] раз
    upsampled = []
    for (item, r) in zip(frame_data, repeat_counts):
        upsampled.extend([item] * r)

    # На всякий случай, проверим длину
    if len(upsampled) != num_frames:
        logger.warning(
            f"Ожидалось {num_frames} кадров после апсемплинга, "
            f"получили {len(upsampled)}. Поправим за счёт обрезки/дублирования."
        )
        if len(upsampled) > num_frames:
            upsampled = upsampled[:num_frames]
        else:
            # если вдруг не хватает, дублируем последний
            while len(upsampled) < num_frames:
                upsampled.append(upsampled[-1])

    return upsampled


def extract_frames(video_path, num_frames=40, method='window'):
    
    # Основная функция: читает видео, выбирает кадры и сохраняет их.

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}.")
   
    frame_sharpness = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        sharpness = calculate_sharpness(frame)
        frame_sharpness.append((frame_count, sharpness))
        frame_count += 1
    
    cap.release()
    
    if not frame_sharpness:
        raise ValueError(f"Список кадров пуст.")

    # 2. Выбираем нужные кадры (Uniform или Window).
    selected_data = get_selected_frame_indices(frame_sharpness, num_frames, method)
    
    if not selected_data:
        raise ValueError(f"Отсутствуют кадры высокой четкости.")

    saved_files = []
    cap = cv2.VideoCapture(video_path) # Открываем заново для сохранения
    
    # selected_data отсортирован хронологически.
    img_set = []
    for (frame_idx, _) in selected_data:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            img_set.append(frame)
        else:
            raise ValueError(f"Сбой чтения кадра {frame_idx}.")
            
    cap.release()

    return img_set



if SOURCE_TYPE == 'frames':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isdir(os.path.join(INPUT_DIR_PATH, item))
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    #dataset_check(items, annotations)
    
    for item in items:          
        input_item_dir_path = INPUT_DIR_PATH / item
        output_item_dir_path = OUTPUT_DIR_PATH / item
    
        img_set = read_images(input_item_dir_path, resize_flag=READ_RESIZE_FLAG)
        
        if len(img_set) != N_FRAMES:
            raise ValueError(f"{len(img_set)} of {N_FRAMES} image files found in '{INPUT_DIR_PATH / item}'.")
        
        if MP_FLAG:
            img_set, annotated_cnt = process_mediapipe(img_set)
            logger.info(f"Размечены {annotated_cnt} из {len(img_set)} кадров ({item}).")

        save_images(img_set, output_item_dir_path, resize_flag=SAVE_RESIZE_FLAG)
    
elif SOURCE_TYPE == 'video':
    items = [
        item for item in os.listdir(INPUT_DIR_PATH)
        if os.path.isfile(os.path.join(INPUT_DIR_PATH, item)) 
        and item.lower().endswith(video_extensions)
        ]
    
    logger.info(f"{len(items)} items found in {INPUT_DIR_PATH}.")
    
    dataset_check(items, annotations)
    
    for item in items:  
        item_name, _ = os.path.splitext(item)
        input_item_dir_path = INPUT_DIR_PATH / item
        output_item_dir_path = OUTPUT_DIR_PATH / item_name
    
        # Cut video into frames.
        img_set = extract_frames(
            video_path=input_item_dir_path,
            num_frames=N_FRAMES,
            method=SELECTION_METHOD
        )

        if len(img_set) != N_FRAMES:
            raise ValueError(f"{len(img_set)} of {N_FRAMES} image files found in '{INPUT_DIR_PATH / item}'.")
        
        if MP_FLAG:
            img_set, annotated_cnt = process_mediapipe(img_set)
            logger.info(f"Размечены {annotated_cnt} из {len(img_set)} кадров ({item}).")

        save_images(img_set, output_item_dir_path, resize_flag=SAVE_RESIZE_FLAG)
else:
    raise ValueError(f"Wrong 'SOURCE_TYPE'.")

'''
# Имя видео без расширения для названия папки (например "video1")
            video_stem = os.path.splitext(file)[0]
            
            current_sharp_output = OUTPUT_SHARP_ROOT / video_stem
            current_mpipe_output = OUTPUT_MPIPE_ROOT / video_stem
            
            
        
            # 2. Запускаем валидацию MediaPipe (если кадры извлеклись)
            if saved_frames:
                frame_functions.process_mediapipe(
                    pictures=saved_frames, 
                    out_dir_name=current_mpipe_output
                )
            
'''