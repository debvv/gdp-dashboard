import streamlit as st
import joblib
import pandas as pd
import numpy as np



try:
    import joblib
    print("joblib успешно импортирован!")
except ModuleNotFoundError as e:
    print(f"Ошибка импорта: {e}")


# Заголовок приложения
st.title("Прогнозирование с использованием моделей машинного обучения")
st.sidebar.header("Настройки")

# Загрузка модели
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model_paths = {
    "Linear Regression": "migration_index_linear_regression_Model.pkl",
    "Random Forest": "migration_index_random_Forest_Model.pkl"
}

# Выбор модели
model_name = st.sidebar.selectbox("Выберите модель", list(model_paths.keys()))
model = load_model(model_paths[model_name])

# Ввод данных для предсказаний
st.header("Ввод данных")
st.write("Введите значения признаков для предсказания.")
input_data = {}

# Пример признаков
features = ["age", "years_experience", "current_salary", "life_quality_index", "political_stability_index"]
for feature in features:
    input_data[feature] = st.number_input(f"Введите значение для {feature}", value=0.0)

# Предсказание
if st.button("Сделать предсказание"):
    input_df = pd.DataFrame([input_data])  # Преобразование в DataFrame
    print("Input DataFrame Columns[1]:", input_df.columns)
    print("Model Expected Features[1]:", model.feature_names_in_)

    prediction = model.predict(input_df)
    st.subheader("Результат предсказания:")
    st.write(prediction)

# Загрузка данных для массовых предсказаний
st.header("Загрузка данных")
uploaded_file = st.file_uploader("Загрузите CSV файл для предсказаний", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:")
    st.write(data)
    print("Input DataFrame Columns[2]:", input_df.columns)
    print("Model Expected Features[2]:", model.feature_names_in_)
    predictions = model.predict(data)
    st.subheader("Результаты предсказаний:")
    st.write(predictions)

# Отображение метрик модели
st.sidebar.header("Информация о модели")
if model_name == "Linear Regression":
    st.sidebar.write("Метрики модели Linear Regression:")
    st.sidebar.write({
        "MAE": 0.0033,
        "MSE": 0.000012,
        "R2": 0.999927
    })
else:
    st.sidebar.write("Метрики модели Random Forest:")
    st.sidebar.write({
        "MAE": 0.0932,
        "MSE": 0.0123,
        "R2": 0.9250
    })

# Логирование (пример)
st.header("Логирование")
if st.button("Сохранить результаты"):
    with open("logs.txt", "a") as log_file:
        log_file.write(f"Ввод: {input_data}, Предсказание: {prediction}\n")
    st.success("Результаты сохранены в лог.")
