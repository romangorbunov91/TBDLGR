import cv2
import numpy as np
import os
import mediapipe as mp
import logging

# Получаем логгер (он настраивается в main.py)
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
    if total_frames <= num_frames:
        logger.warning(f"В видео всего {total_frames} кадров, запрошено {num_frames}. Берем все.")
        return sorted(frame_data, key=lambda x: x[0])

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

def extract_frames(video_path, output_folder, num_frames=40, method='window'):
    """
    Основная функция: читает видео, выбирает кадры и сохраняет их.
    """
    video_name = os.path.basename(video_path)
    logger.info(f"--- Обработка видео: {video_name} ---")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видеофайл: {video_path}.")
        return []

    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Всего кадров ~{total_frames_est}. Метод отбора: {method.upper()}")
    
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
        frame = cv2.resize(frame, (640, 480))
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

def process_mediapipe(pictures, out_dir_name):
    """
    Рисует скелет руки на сохраненных кадрах для валидации.
    """
    if not pictures:
        return

    logger.info(f"--- MediaPipe валидация для {len(pictures)} кадров ---")
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name, exist_ok=True)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        processed_count = 0
        for idx, file in enumerate(pictures):
            if not os.path.exists(file):
                continue

            # Читаем оригинал
            original_image = cv2.imread(file)
            if original_image is None:
                continue
            
            # Флипаем для MP (зеркалирование)
            image = cv2.flip(original_image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                continue

            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Флипаем обратно и сохраняем
            base_name = os.path.basename(file)
            output_filename = os.path.join(out_dir_name, f"annotated_{base_name}")
            cv2.imwrite(output_filename, cv2.flip(annotated_image, 1))
            processed_count += 1
            
        logger.info(f"MediaPipe завершен. Размечено: {processed_count}")