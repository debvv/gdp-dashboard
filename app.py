import streamlit as st
import pandas as pd
import joblib
# Загрузка модели и предварительных настроек
model_paths = {
    "Linear Regression": "migration_index_linear_regression_Model.pkl",
    "Random Forest": "migration_index_random_forest_Model.pkl"
}

scaler_path = "scaler.pkl"  # Убедитесь, что у вас есть сохраненный объект scaler
scaler = joblib.load(scaler_path)

expected_features = ['feature1', 'feature2', 'feature3', ...]  # Список ожидаемых признаков

# Предсказание
def predict_with_model(input_data, model_name):
    # Загрузка модели
    model = joblib.load(model_paths[model_name])
    
    # Подготовка данных
    input_df = pd.DataFrame([input_data])
    
    # Проверка и добавление недостающих признаков
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Или значение по умолчанию
    
    # Переупорядочивание признаков в соответствии с моделью
    input_df = input_df[expected_features]
    
    # Нормализация
    input_df = scaler.transform(input_df)
    
    # Предсказание
    prediction = model.predict(input_df)
    return prediction

# Streamlit интерфейс
import streamlit as st

st.title("Прогнозирование с использованием моделей")
model_name = st.selectbox("Выберите модель", options=model_paths.keys())

st.sidebar.header("Настройки ввода")
input_data = {
    "feature1": st.sidebar.number_input("Feature 1", value=0),
    "feature2": st.sidebar.number_input("Feature 2", value=0),
    # Добавьте остальные признаки
}

if st.button("Предсказать"):
    try:
        prediction = predict_with_model(input_data, model_name)
        st.success(f"Результат предсказания: {prediction}")
    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")