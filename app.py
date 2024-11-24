import streamlit as st
import pandas as pd
import joblib

# Загрузка сохраненных моделей
model_paths = {
    "Linear Regression": "migration_index_linear_regression_Model.pkl",
    "Random Forest": "migration_index_random_Forest_Model.pkl"
}

# Функция для предобработки входных данных
def preprocess_input(input_data, expected_features):
    # Добавление отсутствующих признаков с нулевыми значениями
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    # Перестановка столбцов в соответствии с моделью
    input_data = input_data[expected_features]
    return input_data


# Функция для предсказания
def predict_with_model(model, input_data):
    try:
        # Проверка наличия feature_names_in_ в модели
        if hasattr(model, "feature_names_in_"):
            input_data = preprocess_input(input_data, model.feature_names_in_)
        predictions = model.predict(input_data)
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
    # Добавьте остальные признаки, используемые в модели
}

input_df = pd.DataFrame([input_data])

# Кнопка для предсказания
if st.button("Предсказать"):
    if model:
        prediction = predict_with_model(model, input_df)
        if prediction is not None:
            st.success(f"Предсказание: {prediction}")
    else:
        st.error("Модель не загружена. Выберите модель из списка.")

# Загрузка данных для проверки признаков модели
if model and hasattr(model, "feature_names_in_"):
    st.sidebar.header("Информация о модели")
    st.sidebar.write("Ожидаемые признаки модели:")
    st.sidebar.write(model.feature_names_in_)
