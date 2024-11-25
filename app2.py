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

# Функция для предсказания
def predict_with_model(model, input_data):
    try:
        # Нормализация данных
        if scaler is not None:
            input_data_normalized = scaler.transform(input_data)
        else:
            st.error("Scaler отсутствует. Нормализация не может быть выполнена.")
            return None

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
        st.sidebar.success(f"Модель '{model_name}' загружена.")
    except FileNotFoundError:
        st.sidebar.error(f"Файл модели '{model_paths[model_name]}' не найден.")
        model = None
else:
    st.sidebar.error("Модель не выбрана.")
    model = None

# Ввод данных пользователем
st.header("Введите данные для предсказания")
input_data = {
    "economic_growth_rate": st.number_input("Economic Growth Rate", value=7.1),
    "year": st.number_input("Year", value=2024),
    "total_emigrants": st.number_input("Total Emigrants", value=300000),
    "gdp_per_capita_usd": st.number_input("GDP per Capita (USD)", value=1800),
    "it_growth_potential": st.number_input("IT Growth Potential", value=20095),
    # Добавьте остальные 18 признаков
}

# Преобразование входных данных в DataFrame
input_df = pd.DataFrame([input_data])

# Кнопка для предсказания
if st.button("Предсказать"):
    if model:
        prediction = predict_with_model(model, input_df)
        if prediction is not None:
            st.success(f"Предсказание: {prediction[0]:.2f}")
    else:
        st.error("Модель не загружена. Выберите модель из списка.")
