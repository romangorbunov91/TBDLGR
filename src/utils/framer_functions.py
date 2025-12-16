import os
import numpy as np
import mediapipe as mp
import cv2

# Resize images.
# Prior transforms.
SIZE_INPUT = (360, 270)
# After all transforms.
SIZE_OUTPUT = (224, 224)

# Crop 16:9 verticals.
# READ_RESIZE_FLAG must be True.
CROP_INPUT_FLAG = True

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

def resize_special(img, size, crop_flag=False):
    img_height, img_width, _ = img.shape
    if crop_flag:
        aspect_ratio = img_height / img_width
        if (0.99 * 16/9 <= aspect_ratio <= 1.01 * 16/9):
            # Compute crop bounds: center 50% of height
            top = int(img_height * 0.20)
            bottom = int(img_height * 0.80)
            left = 0
            right = img_width
            return cv2.resize(img[top:bottom, left:right], (size[0], size[0]))
          
    if img_height < img_width:
        return cv2.resize(img, size)
    else:
        return cv2.resize(img, size[::-1])


def read_images(dir_path, resize_flag=False):
    img_names = [
        img for img in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, img)) 
        and img.lower().endswith(image_extensions)
        ]
    
    img_set = []
    for name in img_names:
        img = cv2.imread(os.path.join(dir_path, name))
        if resize_flag:
            img = resize_special(img, SIZE_INPUT, crop_flag=CROP_INPUT_FLAG)
        img_set.append(img)
    
    print(f"Считаны {len(img_set)} кадров {dir_path}")

    return img_set


def save_images(img_set, dir_path, resize_flag=False):
    
    # Create destination folder.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    for idx, img in enumerate(img_set):
        output_filename = os.path.join(dir_path, f"frame_{idx:03d}.jpg")
        if resize_flag:
            success = cv2.imwrite(output_filename, cv2.resize(img, SIZE_OUTPUT))
        else:
            success = cv2.imwrite(output_filename, img)

        if not success:
            raise OSError(f"Ошибка записи: {output_filename}")

    print(f"Сохранены {len(img_set)} кадров {dir_path}")


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
            
            img_flipped = cv2.flip(img, 1)
            
            # Обрабатываем изображение с помощью MediaPipe.
            results = hands.process(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_flipped,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                annotated_cnt += 1
            annotated_img_set.append(cv2.flip(img_flipped, 1))
    
    return annotated_img_set, annotated_cnt


def calculate_sharpness(img):
    # Вычисляет дисперсию Лапласиана (меру резкости).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        print(
            f"В видео всего {total_frames} кадров, запрошено {num_frames}. "
            f"Будет выполнен апсемплинг с дублированием кадров."
        )
        selected_frames = upsample_frame_indices(frame_data, num_frames)
        selected_frames.sort(key=lambda x: x[0])
        return selected_frames

    selected_frames = []

    if method == 'uniform':
        # Равномерное распределение индексов.
        # Например, из 100 кадров берем 0, 25, 50, 75, 99.
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        selected_frames = [frame_data[i] for i in indices]
        print(f"{method}: кадры взяты с равным шагом.")

    elif method == 'window':
        # Делим видео на сегменты и ищем самый резкий кадр в каждом.
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
        
        print(f"{method}: выбраны самые резкие кадры из {num_frames} временных сегментов.")

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
        num_frames: целевое количество кадров (например, 40).
    
    Returns:
        Список длины num_frames, содержащий (frame_index, sharpness_score),
        с возможными повторами, но в хронологическом порядке.
    """
    total_frames = len(frame_data)
    if total_frames > num_frames:
        raise ValueError(f"Неверно указано количество кадров.")
    
    # Гарантируем хронологический порядок исходных кадров.
    frame_data = sorted(frame_data, key=lambda x: x[0])

    # Изначально каждый кадр возьмём по одному разу.
    repeat_counts = np.ones(total_frames, dtype=int)

    # Сколько дублей надо добавить, чтобы получить num_frames.
    extra = num_frames - total_frames  # > 0, так как total < num_frames

    # Равномерно распределим эти дополнительные повторы по временной оси.
    # Идея: линейно интерполируем extra позиций от 0 до total-1.
    extra_indices = np.linspace(0, total_frames - 1, extra, dtype=int)
    # Каждый раз, когда индекс встречается в extra_indices, увеличиваем
    # количество повторов соответствующего кадра на 1
    for idx in extra_indices:
        repeat_counts[idx] += 1

    # Собираем финальный список: кадры идут по времени,
    # каждый дублируется repeat_counts[i] раз.
    upsampled = []
    for (item, r) in zip(frame_data, repeat_counts):
        upsampled.extend([item] * r)

    # На всякий случай, проверим длину
    if len(upsampled) != num_frames:
        raise ValueError(f"Сбой апсемплинга кадров.")

    return upsampled


def extract_frames(video_path, num_frames=40, method='window', resize_flag=False):
    
    # Загружается видео и выбираются.

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

    # Выбираем нужные кадры (Uniform или Window).
    selected_data = get_selected_frame_indices(frame_sharpness, num_frames, method)
    
    if not selected_data:
        raise ValueError(f"Отсутствуют кадры высокой четкости.")

    cap = cv2.VideoCapture(video_path) # Открываем заново для сохранения
    
    # selected_data отсортирован хронологически.
    img_set = []
    for (img_idx, _) in selected_data:
        cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
        ret, img = cap.read()
        if ret:   
            if resize_flag:
                img = resize_special(img, SIZE_INPUT, crop_flag=CROP_INPUT_FLAG)
            img_set.append(img)
        else:
            raise ValueError(f"Сбой чтения кадра {img_idx}.")
            
    cap.release()

    return img_set