import torch
from transformers import AutoTokenizer, AutoModel
import ruyaml as yaml
from scipy.spatial.distance import cosine

# Загрузим предобученную модель и токенайзер
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Функция для преобразования текста в вектор
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Функция для вычисления косинусного расстояния между двумя векторами
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1.numpy(), vec2.numpy())

# Функция для загрузки YAML-файла
def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Функция для сравнения строк и поиска схожих пар
def find_similar_strings(strings, threshold=0.8):
    embeddings = [embed_text(s) for s in strings]
    similar_pairs = []
    
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
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
