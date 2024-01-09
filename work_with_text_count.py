import requests
import pymorphy2 as pm
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import defaultdict, Counter
from razdel import sentenize


page_ids = ['9', '9161687', '91816', '9185409', '89740', '9185932', '9251709', '2892', '6815254']

texts = []

queries = [
    'Россия является главным производителем наиболее опасного из всех асбестовых минералов.',
    'Мыс Спартивенто — это носок итальянского сапога, Санта-Мария-ди-Леука — его каблук.',
    'Солнечное затмение, предсказанное философом, остановило мидо-лидийскую войну.'
]

morph = pm.MorphAnalyzer() # Морфологический анализатор

# Получаем обработанные запросы
queries_list = [[str(morph.parse(word)[0].normal_form) for word in word_tokenize(query, language="russian")] for query in queries]

# Начало обработки текста - получаем тексты из Википедии
for page_id in page_ids:
    response = requests.get(
        'https://ru.wikipedia.org/w/api.php',
        params = {
            'action' : 'query',
            'pageids' : page_id,
            'format' : 'json',
            'prop' : 'extracts',
            'explaintext': True
        }
    ).json()

    texts.append(list(response['query']['pages'].values())[0]["extract"])

text = "".join(texts) # полный текст


sent = sentenize(text)
sentences = [sn.text for sn in sent]

words = []
for s in sentences:
    words.append(word_tokenize(s, language="russian")) # слова

#print(words)

# коллекция с морфологическим анализатором
collection = [[str(morph.parse(word)[0].normal_form) for word in l] for l in words]
ideal_collection = [" ".join(l) for l in collection]
#print(collection)

# Подсчет tf-idf
# df в данном случае - кол-во предложений, в которых встречалось слово
w = list(set([word for sent in collection for word in sent])) # Все слова
id0 = w.index('мыс')
id1 = w.index('спартивенто')



d_df = defaultdict(int)
for word in w:
    # Итерация по предложениям в коллекции
    for sentence in ideal_collection:
        # Проверка, содержит ли предложение текущее слово
        if word in sentence:
            d_df[word] += 1

# df
df_value1 = []
for key, value in d_df.items():
    df_value1.append(value)
#df_value1 = list(d_df.values())

df_value = np.array(df_value1, dtype=np.float32)

# idf
idf_value = np.log10(len(collection) / df_value) # idf


print('idf мыс:',idf_value[id0])
print('idf cпартивенто:', idf_value[id1])

# tf - count (по условию задания)
tf = []

for document in collection:
    term_frequency = Counter(document)
    tf_document = [term_frequency.get(term, 0) for term in w]
    tf.append(tf_document)

tf_value = tf

tf_idf_value = np.multiply(tf_value, idf_value) # tf-idf
norm = [row / np.linalg.norm(row) for row in tf_idf_value] # Нормализация


# Запрос

tf_query = []

for ql in queries_list:
    term_frequency = Counter(ql)
    tf_ql = [term_frequency.get(term, 0) for term in w]
    tf_query.append(tf_ql)

tf_query_value = tf_query


tf_idf_query_value = np.multiply(tf_query_value, idf_value)
norm_query = [row / np.linalg.norm(row) for row in tf_idf_query_value] # Нормализация

# Вывод
num_of_query = 1  # Номер запроса 0, 1 или 2
similarities = [np.dot(norm_query[num_of_query], doc_vector) for doc_vector in norm]
sorted_indices = np.argsort(similarities)[::-1]

qw = 0
for idx in sorted_indices:
    print(f"Вес: {similarities[idx]}, Предложение: {sentences[idx]}\n")
    qw += 1
    if qw == 5:
        break

with open("count_2.txt", "w", encoding="utf-8") as file:
    line = f"Запрос: {queries[num_of_query]}\n"
    file.write(line)

    for idx in sorted_indices:
        line = f"Вес: {similarities[idx]}, Предложение: {sentences[idx]}\n"
        file.write(line)

with open("count_2.txt", "r", encoding="utf-8") as input_file:
    with open("result_count_2.txt", "w", encoding="utf-8") as output_file:
        for line in input_file:
            if "==" in line:
                line1 = input_file.readline()
                line += line1
                output_file.write(line)
            elif line.startswith("Вес") or line.startswith("Запрос"):
                output_file.write(line)
