import numpy as np
import pandas as pd
import torch
import cv2
import streamlit as st
import json
import imgaug.augmenters as iaa
import imageio

from datasets.utils.normalize import normalize

def prepare_frames_for_model(frames):
    clip = list()
    for img in frames:
        clip.append(cv2.resize(img, (224, 224)))

    # [frames, height, width, channels] convert to [height, width, channels, frames].
    clip = np.array(clip).transpose(1, 2, 3, 0)
    # Normalize by 'channels'.
    clip = normalize(clip)

    transforms = iaa.Noop()
    aug_det = transforms.to_deterministic()
    clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)
    # [height, width, channels, frames] convert to [(channels*frames), height, width].
    clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
    clip = clip.float()
    # [(channels*frames), height, width] convert to [batch, (channels*frames), height, width]
    clip = clip.unsqueeze(0)
    return clip

def frames_to_video(frames, output_path, fps=10):
    """
    Сохраняет список кадров (NumPy arrays) в видеофайл.
    frames: список изображений в формате (H, W, C), uint8, BGR или RGB
    output_path: путь для сохранения видео
    fps: кадров в секунду
    """
    
    if not frames:
        raise ValueError("Список кадров пуст")
    # Проверим формат первого кадра
    first_frame = frames[0]
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError("Каждый кадр должен быть (H, W, 3)")
    if first_frame.dtype != 'uint8':
        raise ValueError("Кадры должны быть uint8 (0-255)")

    # Записываем видео с H.264.
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        format='mp4',
        quality=6#,          # 5–9, где 10 — максимальное качество (размер)
        #ffmpeg_params=['-pix_fmt', 'yuv420p']
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

def resize_to_autoplay(img, macro_block_size):
    h, w = img.shape[:2]
    new_w = ((w + macro_block_size - 1) // macro_block_size) * macro_block_size
    new_h = ((h + macro_block_size - 1) // macro_block_size) * macro_block_size
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

@st.cache_resource
def load_model(CONFIG_PATH, MODEL_PATH):
    try:
        from models.temporal import GestureTransformer
        
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        n_classes = config['data']['n_classes']
        device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
        backbone = config['network']['backbone']
        n_head = config['network']['n_head']
        dropout2d = config['network']['dropout2d']
        dropout1d = config['network']['dropout1d']
        ff_size = config['network']['ff_size']
        n_module = config['network']['n_module']
        pretrained = config['network']['pretrained']

        in_planes = 3
        
        model = GestureTransformer(
            backbone=backbone,
            in_planes=in_planes,
            n_classes=n_classes,
            pretrained=pretrained,
            n_head=n_head,
            dropout_backbone=dropout2d,
            dropout_transformer=dropout1d,
            dff=ff_size,
            n_module=n_module
        )
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict']
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        st.success(f"✅ Модель загружена и готова к работе!")
        return model
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def predict_gesture(model, frames, LABEL_MAP_PATH):
    if model is None or len(frames) == 0:
        st.error("Модель не загружена или нет кадров")
        return "Модель не загружена", 0.0, []
    
    # Загрузка текстовых меток.
    label_df = pd.read_csv(LABEL_MAP_PATH)
    # Убедимся, что метки отсортированы по label_encoded.
    label_df = label_df.sort_values('label_encoded')
    class_names = label_df['text'].tolist()
    
    try:
        input_tensor = prepare_frames_for_model(frames)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_conf, top3_idx = torch.topk(probabilities, 3)
            top3_predictions = []
            for i in range(3):
                idx_val = top3_idx[0, i].item()
                conf_val = top3_conf[0, i].item()
                gesture_name = class_names[idx_val] if idx_val < len(class_names) else f"Класс {idx_val}"
                top3_predictions.append((gesture_name, conf_val))
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence_value = confidence.item()
            predicted_idx_value = predicted_idx.item()
        
        if predicted_idx_value < len(class_names):
            predicted_gesture = class_names[predicted_idx_value]
        else:
            predicted_gesture = f"Класс {predicted_idx_value}"
        
        return predicted_gesture, confidence_value, top3_predictions
        
    except Exception as e:
        st.error(f"Ошибка при распознавании: {e}")
        return "Ошибка", 0.0, []