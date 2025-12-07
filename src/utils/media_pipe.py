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
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        cnt = 0
        # Перебираем все изображения.
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

if __name__ == "__main__":
    # Source folder.
    in_folder = 'frames'
    # Destination folder.
    out_folder = 'mediapipe_frames'
          
    for root, dirs, files in os.walk(os.path.join('./datasets/Bukva/', in_folder)):
        out_dir_path = root.replace(in_folder, out_folder)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)  

        media_pipe(root, files, out_dir_path)