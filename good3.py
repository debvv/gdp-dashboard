
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка и подготовка примерного набора данных
def load_dataset():
    # Примерный набор данных (замените на ваш реальный)
    data = {
        "economic_growth_rate": np.random.rand(100),
        "year": np.random.randint(2010, 2025, 100),
        "total_emigrants": np.random.rand(100) * 100000,
        "gdp_per_capita_usd": np.random.rand(100) * 50000,
        "it_growth_potential": np.random.rand(100) * 1000,
        "migration_index": np.random.rand(100) * 100,
        "retention_index": np.random.rand(100) * 50,
        "age_specific_migration_risk": np.random.rand(100) * 10,
        "it_growth_potential": np.random.rand(100) * 70,
    }
    return pd.DataFrame(data)

# Обучение моделей для каждой целевой переменной
def train_models(data, target_columns):
    models = {}
    scaler = MinMaxScaler()
    X = data.drop(columns=target_columns)
    X_scaled = scaler.fit_transform(X)
    for target in target_columns:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Линейная регрессия
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        models[target] = {"Linear Regression": lr_model, "Random Forest": rf_model}
    
    return models, scaler, X.columns

# Обработка входных данных
def preprocess_input(input_data, expected_features, scaler):
    df = pd.DataFrame([input_data])
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Добавление отсутствующих признаков
    df = df[expected_features]
    return scaler.transform(df)

# Streamlit-приложение
st.title("Прогнозирование с использованием моделей машинного обучения")

# Загрузка данных
data = load_dataset()
target_columns = ["migration_index", "retention_index", "age_specific_migration_risk", "it_growth_potential"]
models, scaler, expected_features = train_models(data, target_columns)

# Выбор модели
model_type = st.sidebar.selectbox("Выберите модель", ["Linear Regression", "Random Forest"])
st.sidebar.success(f"Модели готовы для предсказаний!")

# Ввод данных пользователем
st.header("Введите данные для предсказания")
input_data = {
    "economic_growth_rate": st.number_input("Economic Growth Rate", value=7.1),
    "year": st.number_input("Year", value=2024),
    "total_emigrants": st.number_input("Total Emigrants", value=300000),
    "gdp_per_capita_usd": st.number_input("GDP per Capita (USD)", value=1800),
    "it_growth_potential": st.number_input("IT Growth Potential", value=20095),
}

# Проверка и предсказание
if st.button("Предсказать"):
    try:
        processed_input = preprocess_input(input_data, expected_features, scaler)
        predictions = {}
        for target, model_set in models.items():
            model = model_set[model_type]
            prediction = model.predict(processed_input)[0]
            predictions[target] = prediction
        st.success("Предсказания:")
        for target, prediction in predictions.items():
            st.write(f"{target}: {prediction:.2f}")
    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")
