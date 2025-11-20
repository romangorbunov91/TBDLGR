import cv2
import numpy as np
import os


def calculate_sharpness(frame):
    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применяем Лапласов оператор для вычисления резкости
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Вычисляем дисперсию Лапласова оператора как меру резкости
    sharpness = laplacian.var()
    return sharpness


def extract_sharpest_frames(video_path, output_folder, num_frames=40):
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Считываем кадры и вычисляем их резкость
    frame_sharpness = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        sharpness = calculate_sharpness(frame)
        frame_sharpness.append((frame_count, sharpness))
        
        frame_count += 1
    
    cap.release()
    
    # Сортируем кадры по убыванию резкости
    frame_sharpness.sort(key=lambda x: x[1], reverse=True)
    
    # Выбираем топ-40 самых резких кадров
    top_frames = frame_sharpness[:num_frames]
    
    # Сохраняем выбранные кадры
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, (frame_idx, _) in enumerate(top_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_folder, f"sharp_frame_{i+1:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
        cap.release()

    print(f"Извлечено {num_frames} самых резких кадров и сохранено в папку '{output_folder}'.")
    
    # Возвращаем список файлов с изображениями
    return [os.path.join(output_folder, f"sharp_frame_{i+1:03d}.jpg") for i in range(num_frames)]


def media_pipe(pictures, out_dir_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Создаём папку для выходных изображений, если её нет
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    # Инициализация модели MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        # Перебираем все изображения
        for idx, file in enumerate(pictures):
            # Читаем изображение
            image = cv2.flip(cv2.imread(file), 1)
            
            # Обрабатываем изображение с помощью MediaPipe
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            
            # Рисуем landmarks на изображении
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Сохраняем размеченное изображение в выходной папке
            output_filename = os.path.join(out_dir_name, f"annotated_{idx+1}.png")
            cv2.imwrite(output_filename, cv2.flip(annotated_image, 1))
            print(f"Сохранено размеченное изображение: {output_filename}")


# Пример использования
if __name__ == "__main__":
    # Путь к видео
    
    output_folder = 'sharp_frames'  # Папка для сохранения самых резких кадров

    folder_stock_path = "C:\\Users\\ivane\\OneDrive\\Documents\\итмо\\TBDLGR\\trimmed"  # Укажите путь к папке

    for root, dirs, files in os.walk(folder_stock_path):
        for file in files:
            sharp_frames = extract_sharpest_frames(file, output_folder)
