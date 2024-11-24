import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Загрузка тренировочных данных, которые вы использовали при обучении моделей
train_data = pd.read_csv("train_data_final.csv")  # Укажите правильный путь к данным

# Удаление целевых переменных (Y) из данных
target_variables = [
    "migration_index", 
    "retention_index", 
    "it_growth_potential", 
    "age_specific_migration_risk"
]
X_train = train_data.drop(columns=target_variables)

# Создание и обучение Scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# Сохранение Scaler в файл
joblib.dump(scaler, "scaler.pkl")
print("Scaler успешно создан и сохранён в 'scaler.pkl'")
