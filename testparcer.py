import ruyaml as yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Функция для загрузки YAML-файла
def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Функция для сравнения строк и поиска схожих пар
def find_similar_strings(strings, threshold=0.8):
    vectorizer = TfidfVectorizer().fit_transform(strings)
    vectors = vectorizer.toarray()
    
    similar_pairs = []
    cosine_matrix = cosine_similarity(vectors)
    
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            similarity = cosine_matrix[i][j]
            if similarity > threshold:
                similar_pairs.append((strings[i], strings[j], similarity))
    
    return similar_pairs

# Основная функция для обработки списка YAML-файлов
def process_yaml_files(file_paths):
    all_strings = []
    
    for file_path in file_paths:
        data = load_yaml_file(file_path)
        
        # Преобразуем содержимое YAML в список строк
        if isinstance(data, dict):
            all_strings.extend(data.values())
        elif isinstance(data, list):
            all_strings.extend(data)
    
    # Ищем схожие строки
    similar_strings = find_similar_strings(all_strings)
    
    # Выводим результаты
    for str1, str2, similarity in similar_strings:
        print(f"Строка 1: {str1}\nСтрока 2: {str2}\nСхожесть: {similarity:.2f}\n")

# Пример использования
file_paths = ["file1.yml", "file2.yml"]
process_yaml_files(file_paths)
