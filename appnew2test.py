import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Генерация демонстрационного набора данных
def generate_demo_data():
    X, y = make_regression(n_samples=500, n_features=23, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")
# Создание моделей
def create_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
# Интерфейс Streamlit
st.title("Динамическое создание и предсказание моделей")



# Генерация данных
st.sidebar.header("Настройки данных")
X, y = generate_demo_data()
train_size = st.sidebar.slider("Размер тренировочной выборки (%)", 10, 90, 80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_size) / 100, random_state=42)

# Выбор модели
model_name = st.sidebar.selectbox("Выберите модель", ["Linear Regression", "Random Forest"])
models = create_models()
model = models[model_name]

# Обучение модели
st.sidebar.subheader("Обучение модели")
train_button = st.sidebar.button("Обучить модель")
if train_button:
    model.fit(X_train, y_train)
    st.sidebar.success(f"Модель '{model_name}' обучена.")

# Ввод данных пользователем
st.header("Введите данные для предсказания")
input_data = {f"feature_{i}": st.number_input(f"Feature {i}", value=0.0) for i in range(X.shape[1])}
input_df = pd.DataFrame([input_data])

# Отображение данных
st.write("Данные для предсказания:")
st.write(input_df)

# Предсказание
predict_button = st.button("Предсказать")
if predict_button:
    try:
        prediction = model.predict(input_df)
        st.success(f"Предсказание: {prediction}")
    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")
# Метрики модели
if train_button and st.sidebar.checkbox("Показать метрики модели"):
    y_pred = model.predict(X_test)
    st.subheader("Метрики модели")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.write(f"R2: {r2_score(y_test, y_pred):.4f}")

# Загрузка сохраненных моделей
#model_paths = {
 #   "Linear Regression": "migration_index_linear_regression_Model.pkl",
  #  "Random Forest": "migration_index_random_Forest_Model.pkl"
#}

# Функция для предобработки входных данных
#def preprocess_input(input_data, expected_features):
    # Добавление отсутствующих признаков с нулевыми значениями
 #   for feature in expected_features:
  #      if feature not in input_data.columns:
   #         input_data[feature] = 0
    # Перестановка столбцов в соответствии с моделью
    #input_data = input_data[expected_features]
    #return input_data

# Функция для предсказания
#def predict_with_model(model, input_data):
 #   try:
        # Проверка наличия feature_names_in_ в модели
  #      if hasattr(model, "feature_names_in_"):
   #         input_data = preprocess_input(input_data, model.feature_names_in_)
    #    predictions = model.predict(input_data)
     #   return predictions
    #except Exception as e:
     #   st.error(f"Ошибка во время предсказания: {e}")
      #  return None

# Интерфейс Streamlit
#st.title("Прогнозирование с использованием моделей машинного обучения")

# Выбор модели
#model_name = st.sidebar.selectbox("Выберите модель", list(model_paths.keys()))

# Загрузка модели
#if model_name in model_paths:
 #   try:
  #      model = joblib.load(model_paths[model_name])
   #     st.sidebar.success(f"Модель '{model_name}' загружена.")
    #except FileNotFoundError:
     #   st.sidebar.error(f"Файл модели '{model_paths[model_name]}' не найден.")
      #  model = None
#else:
 #   st.sidebar.error("Модель не выбрана.")
  #  model = None

# Ввод данных пользователем
#st.header("Введите данные для предсказания")
#input_data = {
 #   "economic_growth_rate": st.number_input("Economic Growth Rate", value=7.1),
  #  "year": st.number_input("Year", value=2024),
   # "total_emigrants": st.number_input("Total Emigrants", value=300000),
   # "gdp_per_capita_usd": st.number_input("GDP per Capita (USD)", value=1800),
   # "it_growth_potential": st.number_input("IT Growth Potential", value=20095),
   # "difference_in_share": st.number_input("Difference in Share", value=1.5),
  #  "it_sector_investments": st.number_input("IT Sector Investments (USD)", value=50000),
  #  "specialist_id": st.number_input("Specialist ID (for internal use)", value=0),
  #  "net_migration_persons": st.number_input("Net Migration Persons", value=20000),
  #  "life_quality_index": st.number_input("Life Quality Index", value=75.4),
  #  "age": st.number_input("Age", value=30),
 #   "inflation_rate": st.number_input("Inflation Rate (%)", value=3.2),
  #  "years_experience": st.number_input("Years of Experience", value=5),
  #  "unemployment_rate": st.number_input("Unemployment Rate (%)", value=4.5),
  #  "internet_access": st.number_input("Internet Access (%)", value=85),
 #   "retention_index": st.number_input("Retention Index", value=0.8),
  #  "political_stability_index": st.number_input("Political Stability Index", value=60),
  #  "current_salary": st.number_input("Current Salary (USD)", value=50000),
 #   "migration_index": st.number_input("Migration Index", value=0.7),
 #   "age_specific_migration_risk": st.number_input("Age Specific Migration Risk", value=0.2),
  #  "percentage_in_population": st.number_input("Percentage in Population (%)", value=5.3),
  #  "total_international_migrants": st.number_input("Total International Migrants", value=100000),
  #  "average_it_salary": st.number_input("Average IT Salary (USD)", value=80000)
#}

#input_df = pd.DataFrame([input_data])

# Проверка наличия NaN
#if input_df.isnull().any().any():
  #  st.error("Некоторые признаки имеют пропущенные значения. Проверьте ввод.")

# Вывод данных для предсказания
#st.write("Данные для предсказания:")
#st.write(input_df)

# Применение scaler (если необходимо)
#if model and hasattr(model, "feature_names_in_"):
 #   st.sidebar.write("Ожидаемые признаки модели:")
  #  st.sidebar.write(model.feature_names_in_)
    # Проверка наличия scaler
 #   try:
 #       scaler = joblib.load("scaler.pkl")
  #      input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
  #  except FileNotFoundError:
 #       st.warning("Scaler не найден. Используются необработанные данные.")
  #      input_df_scaled = input_df
#else:
 #   input_df_scaled = input_df

# Кнопка для предсказания
#if st.button("Предсказать"):
  #  if model:
   #     prediction = predict_with_model(model, input_df_scaled)
  #      if prediction is not None:
   #         st.success(f"Предсказание: {prediction}")
  #  else:
  #      st.error("Модель не загружена. Выберите модель из списка.")
