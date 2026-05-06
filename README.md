# Проект классификации изображений

## Состав проекта

- `main.py` - приложение FastAPI для классификации изображений.
- `app.py` - интерфейс Streamlit.
- `best_classification_model.keras` - лучшая модель по F1-мере.
- `best_model_metadata.json` - описание модели, классов и входной формы.
- `model_comparison_metrics.csv` - сводная таблица сравнения моделей.
- `requirements.txt` - зависимости проекта.
- `Dockerfile` - контейнер для развёртывания API.

## Лучшая модель

- Модель: MobileNetV2 transfer learning для набора кошек, собак и обезьян
- Датасет: custom_3class
- Accuracy: 0.9884
- Recall: 0.9895
- Precision: 0.9890
- F1: 0.9892
- Среднее время инференса на одно изображение: 0.000606 с

## Локальный запуск API

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Локальный запуск интерфейса

```bash
streamlit run app.py
```

## Пример запроса

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@example.png"
```
