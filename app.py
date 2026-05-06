import io
import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title='Классификация изображений', layout='centered')
st.title('Классификация изображений')
API_URL = st.text_input('Адрес API', 'http://127.0.0.1:8000/predict')
source = st.radio('Источник изображения', ['Загрузка файла'])
image = None

if source == 'Загрузка файла':
    uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

if image is not None:
    st.image(image, caption='Входное изображение', use_container_width=True)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    if st.button('Классифицировать'):
        files = {'file': ('image.png', buffer.getvalue(), 'image/png')}
        response = requests.post(API_URL, files=files, timeout=60)
        if response.ok:
            result = response.json()
            st.success(f"Предсказанный класс: {result['predicted_class']}")
            st.bar_chart(result.get('probabilities', {}))
            st.json(result)
        else:
            st.error(f'Ошибка API: {response.status_code}')
            st.text(response.text)
