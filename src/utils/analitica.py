import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'src/utils/annotations.csv'  
df = pd.read_csv(file_path)

# Оставляем только нужные столбцы
df = df[['label_encoded', 'split']]

# Анализ столбца 'label_encoded'
label_encoded_analysis = df['label_encoded'].describe()  # Статистика для меток
label_encoded_unique_count = df['label_encoded'].nunique()  # Количество уникальных значений в label_encoded
label_encoded_value_counts = df['label_encoded'].value_counts()  # Частота уникальных значений

# Анализ столбца 'split'
split_analysis = df['split'].describe()  # Статистика для split
split_unique_count = df['split'].nunique()  # Количество уникальных значений в split
split_value_counts = df['split'].value_counts()  # Частота уникальных значений
split_percentage = df['split'].value_counts(normalize=True) * 100  # Процентное соотношение

# Выводим результаты
print(f"\nАнализ 'label_encoded':\n{label_encoded_analysis}")
print(f"\nКоличество уникальных меток: {label_encoded_unique_count}")
print(f"\nЧастотность меток:\n{label_encoded_value_counts}")

print(f"\nАнализ 'split':\n{split_analysis}")
print(f"\nКоличество уникальных значений в 'split': {split_unique_count}")
print(f"\nЧастотность значений 'split':\n{split_value_counts}")
print(f"\nПроцентное соотношение значений 'split':\n{split_percentage}")

# Визуализация

# График для столбца 'label_encoded'
plt.figure(figsize=(8, 5))
label_encoded_value_counts.plot(kind='bar', color='skyblue')
plt.title('Распределение меток в label_encoded')
plt.xlabel('Метка')
plt.ylabel('Частота')
plt.xticks(rotation=0)
plt.show()

# График для столбца 'split' - частота
plt.figure(figsize=(8, 5))
split_value_counts.plot(kind='bar', color='lightgreen')
plt.title('Распределение значений в split')
plt.xlabel('Split')
plt.ylabel('Частота')
plt.xticks(rotation=0)
plt.show()

# График для столбца 'split' - процентное соотношение
plt.figure(figsize=(8, 5))
split_percentage.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Процентное соотношение значений в split')
plt.ylabel('')  # Убираем метку y
plt.show()




