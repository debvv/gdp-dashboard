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
# Функция для обработки признаков
def preprocess_input(input_data, model_features):
    """
    Приведение входных данных в соответствие с признаками, которые использовались для обучения модели.
    """
    # Добавить недостающие признаки с нулевыми значениями
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Или любое значение по умолчанию (например, среднее из тренировочного набора)

    # Удалить лишние признаки, которых нет в модели
    input_data = input_data[model_features]
    
    return input_data

# Функция для предсказания
def predict_with_model(model, input_data):
    try:
        # Проверить ожидаемые признаки модели
        if hasattr(model, "feature_names_in_"):
            input_data = preprocess_input(input_data, model.feature_names_in_)
        
        # Нормализация данных
        if scaler:
            input_data_normalized = scaler.transform(input_data)
        else:
            input_data_normalized = input_data

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

# Ввод данных пользователем
st.header("Введите данные для предсказания")
input_data = {
    "economic_growth_rate": st.number_input("Economic Growth Rate", value=7.1),
    "year": st.number_input("Year", value=2010),
    "total_emigrants": st.number_input("Total Emigrants", value=300000),
    "gdp_per_capita_usd": st.number_input("GDP per Capita (USD)", value=1800),
    "it_growth_potential": st.number_input("IT Growth Potential", value=20095),
    # Добавьте все остальные признаки (например, age, inflation_rate, etc.)
    "age": st.number_input("Age", value=30),
    "average_it_salary": st.number_input("Average IT Salary (USD)", value=80000),
    "current_salary": st.number_input("Current Salary (USD)", value=50000),
    "difference_in_share": st.number_input("Difference in Share", value=1.5),
    "inflation_rate": st.number_input("Inflation Rate (%)", value=3.2),
    "years_experience": st.number_input("Years of Experience", value=5),
    "unemployment_rate": st.number_input("Unemployment Rate (%)", value=4.5),
    "internet_access": st.number_input("Internet Access (%)", value=85),
    "retention_index": st.number_input("Retention Index", value=0.8),
    "political_stability_index": st.number_input("Political Stability Index", value=60),
    "net_migration_persons": st.number_input("Net Migration Persons", value=20000),
    "migration_index": st.number_input("Migration Index", value=0.7),
    "age_specific_migration_risk": st.number_input("Age Specific Migration Risk", value=0.2),
    "percentage_in_population": st.number_input("Percentage in Population (%)", value=5.3),
    "total_international_migrants": st.number_input("Total International Migrants", value=100000)
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

