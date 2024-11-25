import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Загрузка сохраненных моделей
model_paths = {
    "Linear Regression": "migration_index_linear_regression_Model.pkl",
    "Random Forest": "migration_index_random_Forest_Model.pkl"
}

# Загрузка scaler
scaler_path = "scaler.pkl"  # Убедитесь, что этот файл существует в проекте
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Scaler не найден. Убедитесь, что scaler.pkl загружен.")
    scaler = None
    


# Функция для обработки данных
def preprocess_input(input_data, model_features):
    """
    Приведение входных данных в соответствие с признаками, которые использовались для обучения модели.
    """
    # Добавить недостающие признаки с нейтральным значением (например, 0)
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Можно заменить на среднее значение признака из тренировочного набора
    
    # Удалить лишние признаки, которых нет в обученной модели
    input_data = input_data[model_features]
    return input_data

# Функция для предсказания
def predict_with_model(model, input_data):
    try:
        # Проверить, есть ли информация о признаках модели
        if hasattr(model, "feature_names_in_"):
            input_data = preprocess_input(input_data, model.feature_names_in_)
        
        # Нормализация данных, если используется MinMaxScaler
        if scaler:
            input_data_normalized = scaler.transform(input_data)
        else:
            input_data_normalized = input_data
        
        # Выполнить предсказание
        predictions = model.predict(input_data_normalized)
        return predictions
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
# Преобразование входных данных в DataFrame
input_df = pd.DataFrame([input_data])


# Вывод данных
st.write("Данные для предсказания:")
st.write(input_df)

# Предсказание
if st.button("Предсказать"):
    if model:
        prediction = predict_with_model(model, input_df)
        if prediction is not None:
            st.success(f"Предсказание: {prediction}")
    else:
        st.error("Модель не загружена. Выберите модель из списка.")
