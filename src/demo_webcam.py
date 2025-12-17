import streamlit as st
import cv2
import torch
import numpy as np
import time
from pathlib import Path
import sys
import pandas as pd
from datasets.utils.normalize import normalize
import imgaug.augmenters as iaa
from utils.framer_functions import resize_special

sys.path.insert(0, str(Path(__file__).parent / "src"))
MODEL_PATH = r".\checkpoints\Bukva\best_train_bukva.pth"

st.title("✌️ Распознаватель РЖЯ (Дактиль) - Режим веб-камеры")
st.markdown("""
Нажмите "Начать запись", и система захватит 40 последовательных кадров для анализа.
""")

with st.expander("ℹ️ Как это работает", expanded=False):
    st.markdown("""
    1. **Нажмите "Начать запись"** - запустится обратный отсчет
    2. **Подготовьте жест** - в течение 3 секунд приготовьтесь показать жест
    3. **Система захватит 40 кадров** с веб-камеры
    4. **Кадры обрабатываются** (изменение размера, нормализация)
    5. **Модель анализирует последовательность кадров** и распознаёт жест
    6. **Отображается результат** с указанием уровня уверенности
    """)

if 'captured_frames' not in st.session_state:
    st.session_state.captured_frames = []
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = None
if 'top3_result' not in st.session_state:
    st.session_state.top3_result = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'show_countdown' not in st.session_state:
    st.session_state.show_countdown = False

def capture_frames_from_camera(num_frames=40):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error("Не удалось открыть веб-камеру!")
        return []
    
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    #for _ in range(num_frames):
    #    cap.read()
    
    st.info(f"Захватываем {num_frames} кадров с веб-камеры...")
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = resize_special(img, (360, 270), crop_flag=True)
            frames.append(img)
        else:
            raise ValueError(f"Сбой чтения кадра.")
        
        #time.sleep(0.05)
    
    cap.release()
    
    if frames:
        st.success(f"Захвачено {len(frames)} кадров")
    
    return frames

@st.cache_resource
def load_model():
    try:
        from models.temporal import GestureTransformer
        import json
        with open('src/hyperparameters/Bukva/config.json', 'r') as f:
            config = json.load(f)
        
        backbone = config['network']['backbone']
        n_classes = config['data']['n_classes']
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict']
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        st.success(f"✅ Модель загружена успешно!")
        return model
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def prepare_frames_for_model(frames):
    clip = list()
    for frame in frames:
        resized = cv2.resize(frame, (224, 224))
        clip.append(resized)
    clip = np.array(clip).transpose(1, 2, 3, 0)
    clip = normalize(clip)
    transforms = iaa.Noop()
    aug_det = transforms.to_deterministic()
    clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)
    clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
    clip = clip.float()
    clip = clip.unsqueeze(0)
    return clip


def predict_gesture(model, frames):
    if model is None or len(frames) == 0:
        st.error("Модель не загружена или нет кадров")
        return "Модель не загружена", 0.0, []
    
    # --- Загрузка текстовых меток ---
    label_mapping_path = "./src/datasets/bukva_label_mapping.csv"  # или укажите полный путь, если нужно
    label_df = pd.read_csv(label_mapping_path)
    # Убедимся, что метки отсортированы по label_encoded
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

st.sidebar.header("📷 Запись с веб-камеры")

st.sidebar.info("""
1. Нажмите кнопку "Начать запись"
2. Подготовьтесь к жесту
3. Система захватит 40 кадров
4. Получите результат распознавания
""")

if st.sidebar.button("🎬 Начать запись с камеры", type="primary", 
                     disabled=st.session_state.is_recording,
                     use_container_width=True):
    st.session_state.is_recording = True
    st.session_state.show_countdown = True
    st.session_state.captured_frames = []
    st.session_state.recognition_result = None
    st.session_state.top3_result = None

if st.session_state.show_countdown:
    countdown_placeholder = st.empty()
    for i in range(3, 0, -1):
        countdown_placeholder.warning(f"⏳ Подготовьтесь! Запись начнется через {i}...")
        time.sleep(1)
    
    countdown_placeholder.info("📹 Идет запись...")
    st.session_state.show_countdown = False
    
    with st.spinner("Захватываем кадры с веб-камеры..."):
        frames = capture_frames_from_camera(40)
        
        if frames:
            st.session_state.captured_frames = frames
            st.session_state.is_recording = False
            
            st.subheader("📷 Захваченные кадры с веб-камеры")
            cols = st.columns(4)
            for idx, frame in enumerate(frames):
                with cols[idx % 4]:
                    st.image(frame, caption=f"Кадр {idx+1}", width=150)
            
            if 'model' not in st.session_state:
                with st.spinner("Загружаем модель распознавания..."):
                    st.session_state.model = load_model()
            
            if st.session_state.model:
                with st.spinner("🤖 Распознаем жест..."):
                    gesture, confidence, top3_list = predict_gesture(st.session_state.model, frames)
                    st.session_state.recognition_result = (gesture, confidence)
                    st.session_state.top3_result = top3_list
                    
                    st.success(f"""
                    ## 🎯 Результат распознавания
                    ### **Жест: {gesture}**
                    Уверенность: {confidence:.1%}
                    """)
                    
                    if top3_list:
                        st.subheader("🏆 Топ-3 альтернативных варианта")
                        cols = st.columns(3)
                        for i, (name, prob) in enumerate(top3_list):
                            with cols[i]:
                                medal_color = ["#FFD700", "#C0C0C0", "#CD7F32"][i]
                                st.markdown(f"<h4 style='text-align: center; color: {medal_color};'>{i+1} место</h4>", unsafe_allow_html=True)
                                st.markdown(f"<h3 style='text-align: center;'>{name}</h3>", unsafe_allow_html=True)
                                st.progress(float(prob))
                                st.markdown(f"<p style='text-align: center;'>{prob:.1%}</p>", unsafe_allow_html=True)
                    else:
                        st.info("Не удалось получить альтернативные варианты.")
            else:
                st.error("Модель не загружена!")
        else:
            st.error("Не удалось захватить кадры с веб-камеры")
            st.session_state.is_recording = False

if st.session_state.recognition_result:
    gesture, confidence = st.session_state.recognition_result
    st.sidebar.subheader("📊 Результат")
    st.sidebar.metric("Распознанный жест", gesture)
    st.sidebar.metric("Уверенность", f"{confidence:.1%}")

if 'model' not in st.session_state:
    with st.spinner("Загружаем модель распознавания..."):
        st.session_state.model = load_model()

st.sidebar.divider()
if st.session_state.is_recording:
    st.sidebar.warning("🔄 Идет запись...")
else:
    st.sidebar.info("✅ Готов к записи")

st.divider()
st.caption("Система распознавания РЖЯ (Дактиль) | Режим веб-камеры")