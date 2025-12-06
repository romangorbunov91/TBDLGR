import numpy as np
import os
import mediapipe as mp
import cv2

def media_pipe(root, files, out_dir_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Инициализация модели MediaPipe Hands.
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        cnt = 0
        # Перебираем все изображения
        for idx, file in enumerate(files):
            
            # Читаем изображение.
            image = cv2.flip(cv2.imread(os.path.join(root, file)), 1)
            
            # Обрабатываем изображение с помощью MediaPipe.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                       
            # Сохраняем размеченное изображение в выходной папке.
            output_filename = os.path.join(out_dir_path, f"frame_{idx:03d}.jpg")
            annotated_image = image.copy()
            
            if not results.multi_hand_landmarks:
                cv2.imwrite(output_filename, cv2.flip(image, 1))
            # Рисуем landmarks на изображении.
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                annotated_image_resized = cv2.resize(annotated_image, (224, 224))
                cv2.imwrite(output_filename, cv2.flip(annotated_image_resized, 1))
            cnt += 1
        
        print(f"Сохранено {cnt} кадров: {out_dir_path}")

# Пример использования.

if __name__ == "__main__":
    # Папка с исходными изображениями
    in_folder = 'photo'
    
    # Папка для сохранения обработанных изображений
    out_folder = 'mediapipe_frames'
    
    # Получаем текущую директорию
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к папке с исходными изображениями
    photos_path = os.path.join(current_dir, in_folder)
    
    # Путь для выходных данных
    out_path = os.path.join(current_dir, out_folder)

    # Проверяем, существует ли папка 'photos'
    if not os.path.exists(photos_path):
        print(f"Ошибка: Папка {photos_path} не найдена!")
    else:
        print(f"Папка с изображениями найдена: {photos_path}")
    
    # Перебираем все файлы в папке 'photos'
    for root, dirs, files in os.walk(photos_path):
        # Создаем выходную директорию для каждого подкаталога
        out_dir_path = root.replace(photos_path, out_path)
        
        # Проверяем, существует ли папка для выходных данных
        if not os.path.exists(out_dir_path):
            print(f"Создаём папку: {out_dir_path}")
            os.makedirs(out_dir_path)  

        # Вызываем функцию обработки
        media_pipe(root, files, out_dir_path)
