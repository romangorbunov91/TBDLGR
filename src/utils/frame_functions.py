import cv2
import numpy as np
import os
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

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

def extract_frames(video_path, output_folder, num_frames=40, method='window'):
    """
    Основная функция: читает видео, выбирает кадры и сохраняет их.
    """

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}.")
        return []
   
    # 1. Считываем метаданные всех кадров
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
        logger.warning("Список кадров пуст.")
        return []

    # 2. Выбираем нужные кадры (Uniform или Window)
    selected_data = get_selected_frame_indices(frame_sharpness, num_frames, method)
    
    if not selected_data:
        return []
        
    logger.info(f"Диапазон индексов: {selected_data[0][0]} -> {selected_data[-1][0]}")
    
    # 3. Сохраняем кадры на диск
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    saved_files = []
    cap = cv2.VideoCapture(video_path) # Открываем заново для сохранения
    
    # selected_data отсортирован хронологически.
    # i - порядковый номер (0, 1, 2...), определяющий порядок сохранения
    for i, (frame_idx, score) in enumerate(selected_data):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (224, 224))
        if ret:
            # Имя файла: frame_000.jpg, frame_001.jpg... (гарантирует хронологию)
            frame_filename = os.path.join(output_folder, f"frame_{i:03d}.jpg")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                saved_files.append(frame_filename)
            else:
                logger.error(f"Ошибка записи: {frame_filename}")
        else:
            logger.warning(f"Сбой чтения кадра {frame_idx} при сохранении")
            
    cap.release()

    logger.info(f"Сохранено {len(saved_files)} кадров в '{output_folder}'.")
    return saved_files

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
    
    logger.info(f"Размечены {annotated_cnt} из {len(img_set)} кадров.")
    return annotated_img_set


def read_images(files, root):
    img_set = []
    for file in files:
        img_set.append(cv2.flip(cv2.imread(os.path.join(root, file)), 1))
    return img_set


def save_images(img_set, out_dir_path, resize_flag=False):
    for idx, img in enumerate(img_set):
        output_filename = os.path.join(out_dir_path, f"frame_{idx:03d}.jpg")
        if resize_flag:
            cv2.imwrite(output_filename, cv2.flip(cv2.resize(img, (224, 224)), 1))
        else:
            cv2.imwrite(output_filename, cv2.flip(img, 1))
    
    logger.info(f"Сохранено {len(img_set)} кадров {out_dir_path}")

