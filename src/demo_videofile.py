import cv2
import streamlit as st
import json
import tempfile
import os

from utils.framer import extract_frames
from demo_functions import load_model, predict_gesture, frames_to_video

MODEL_PATH = r".\checkpoints\Bukva\best_train_bukva.pth"
CONFIG_PATH = r".\src\hyperparameters\Bukva\config.json"
LABEL_MAP_PATH = r".\src\datasets\bukva_label_map.csv"

st.title("✌️ Распознаватель РЖЯ (Дактиль) - Режим видеофайла")
st.markdown("""
Загрузите видеофайл
""")

if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'extracted_frames' not in st.session_state:
    st.session_state.extracted_frames = []
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = None
if 'top3_result' not in st.session_state:
    st.session_state.top3_result = None

st.sidebar.header("📁 Загрузка видео")

uploaded_file = st.sidebar.file_uploader(
    "Выберите видеофайл (MP4, AVI, MOV)",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv']
)

import base64
def autoplay_video(video_path):
    """
    Воспроизводит видео автоматически (без звука, с зацикливанием по желанию)
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    
    video_html = f"""
    <video autoplay muted loop playsinline style="width: 100%; max-width: 600px; height: auto;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Ваш браузер не поддерживает видео.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

if uploaded_file is not None:
    temp_video_path = f"temp_{uploaded_file.name}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.uploaded_video = temp_video_path
    
    st.sidebar.success(f"Файл загружен: {uploaded_file.name}")
    
    if st.sidebar.button("🎬 Распознать жест", type="primary"):
        with st.spinner("Извлекаем кадры из видео..."):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                n_frames = config['data']['n_frames']
            
            frames_RGB = extract_frames(
                video_path=temp_video_path,
                num_frames=n_frames,
                method='window',
                resize_flag=True
            )
            frames = []
            for frame in frames_RGB:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frames:
                st.session_state.extracted_frames = frames
                
                with st.expander("📷 Извлечённые кадры из видео"):
                    cols = st.columns(10)
                    for idx, frame in enumerate(frames):
                        with cols[idx % 10]:
                            st.image(frame, caption=f"Кадр {idx}", width=50)
                
                with st.spinner("🎥 Создаём видео из кадров..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                        frame_video_path = tmpfile.name
                    try:
                        frames_to_video(frames, frame_video_path, fps=20)

                        if not os.path.exists(frame_video_path):
                            st.error("Видео не создано")
                            
                        if os.path.getsize(frame_video_path) == 0:
                            st.error("Видео пустое (0 байт)")
                            
                        #with open(frame_video_path, "rb") as f:
                        #    st.video(f.read())
                        autoplay_video(frame_video_path)
                    
                    finally:
                        # Удаляем временный файл
                        os.unlink(frame_video_path)
                        if os.path.exists(frame_video_path):
                            os.unlink(frame_video_path)
                
                if st.session_state.model:
                    with st.spinner("🤖 Распознаем жест..."):
                        gesture, confidence, top3_list = predict_gesture(
                            st.session_state.model,
                            frames,
                            LABEL_MAP_PATH
                            )
                        st.session_state.recognition_result = (gesture, confidence)
                        st.session_state.top3_result = top3_list
                        
                        st.success(f"""
                        ## 🎯 **Буква: {gesture}**
                        Уверенность: {confidence:.1%}
                        """)
                        
                        if top3_list:
                            st.subheader("🏆 Топ-3 по уверенности")
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
                st.error("Не удалось извлечь кадры из видео")

if st.session_state.recognition_result:
    gesture, confidence = st.session_state.recognition_result
    st.sidebar.subheader("📊 Результат")
    st.sidebar.metric("Буква", gesture)
    st.sidebar.metric("Уверенность", f"{confidence:.1%}")

with st.expander("ℹ️ Как это работает"):
    st.markdown("""
    1. **Загрузите видеофайл** с жестом (MP4, AVI, MOV и другие форматы)
    2. **Система извлечёт кадры** из видео
    3. **Кадры обрабатываются** (изменение размера, нормализация)
    4. **Модель анализирует последовательность кадров** и распознаёт жест
    5. **Отображается результат** с указанием уровня уверенности
    """)

if 'model' not in st.session_state:
    with st.spinner("Загружаем модель распознавания..."):
        st.session_state.model = load_model(CONFIG_PATH, MODEL_PATH)

st.divider()
st.caption("Система распознавания РЖЯ (Дактиль) | Режим видеофайла")