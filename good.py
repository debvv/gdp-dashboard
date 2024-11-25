import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Загрузка сохранённых моделей
model_paths = {
    "Linear Regression": "migration_index_linear_regression_Model.pkl",
    "Random Forest": "migration_index_random_forest_Model.pkl"
}

# Коэффициент для коррекции предсказания
CORRECTION_FACTOR = 0.01383

# Функция для обработки данных
def preprocess_input(input_data, model_features):
    """
    Приведение входных данных в соответствие с признаками, которые использовались для обучения модели.
    """
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Добавить отсутствующие признаки
    input_data = input_data[model_features]  # Удалить лишние признаки
    return input_data

# Функция для предсказания
def predict_with_model(model, input_data):
    try:
        # Проверить, есть ли информация о признаках модели
        if hasattr(model, "feature_names_in_"):
            input_data = preprocess_input(input_data, model.feature_names_in_)
        
        # Выполнить предсказание
        predictions = model.predict(input_data)
        
        # Применить коэффициент коррекции
        predictions_corrected = predictions * CORRECTION_FACTOR
        return predictions_corrected
    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")
        return None

# Интерфейс Streamlit
st.title("Прогнозирование с использованием моделей машинного обучения")

# Выбор модели
model_name = st.sidebar.selectbox("Выберите модель", list(model_paths.keys()))

# Загрузка модели
if model_name in model_paths:
    try:
        model = joblib.load(model_paths[model_name])
        st.sidebar.success(f"Модель '{model_name}' готова для предсказаний!")
    except FileNotFoundError:
        st.sidebar.error(f"Файл модели '{model_paths[model_name]}' не найден.")
        model = None

# Ввод данных пользователем
st.header("Введите данные для предсказания")
input_data = {
    "economic_growth_rate": st.number_input("Economic Growth Rate", value=7.1),
    "year": st.number_input("Year", value=2024),
    "total_emigrants": st.number_input("Total Emigrants", value=300000),
    "gdp_per_capita_usd": st.number_input("GDP per Capita (USD)", value=1800),
    "it_sector_investments": st.number_input("IT Sector Investments (USD)", value=50000),
    "life_quality_index": st.number_input("Life Quality Index", value=75.4),
    "specialist_id": st.number_input("Specialist ID (for internal use)", value=0)
}

input_df = pd.DataFrame([input_data])

# Вывод данных
st.write("Данные для предсказания:")
st.write(input_df)

# Предсказание
if st.button("Предсказать"):
    if model:
        prediction = predict_with_model(model, input_df)
        if prediction is not None:
            st.success(f"Скорректированное предсказание: {prediction[0]:.2f}")
    else:
        st.error("Модель не загружена. Выберите модель из списка.")
