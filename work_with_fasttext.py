import requests
import pymorphy2 as pm
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import defaultdict, Counter
from razdel import sentenize
import fasttext

nltk.download('punkt')

model = fasttext.load_model("wiki.ru.bin") # предобученная модель


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

collection = [[str(morph.parse(word)[0].normal_form) for word in l] for l in words]
vectors = [np.mean([model.get_word_vector(word) for word in s], axis=0) if s else np.zeros(model.get_dimension()) for s in collection] # Векторы предложений

# Косинусная близость
similarities = [
    {
        sentences[i]: (
            np.dot(vector, np.mean([model.get_word_vector(word) for word in query], axis=0))
            / (np.linalg.norm(np.mean([model.get_word_vector(word) for word in query], axis=0)) * np.linalg.norm(vector)) if np.linalg.norm(vector) else 0
        )
        for i, vector in enumerate(vectors)
    }
    for query in queries_list
]

# Первые 5 самых релевантных предложений для каждого запроса
ok = 0
for _, w in enumerate(similarities):
  sorted_indices = sorted(w, key=w.get, reverse=True)
  qw = 0
  if ok == 0:
    print('=======')
    print('Запрос: Россия является главным производителем наиболее опасного из всех асбестовых минералов.')
    print('=======')
    print()
  elif ok == 1:
    print('=======')
    print('Запрос: Мыс Спартивенто — это носок итальянского сапога, Санта-Мария-ди-Леука — его каблук.')
    print('=======')
    print()
  elif ok == 2:
    print('=======')
    print('Запрос: Солнечное затмение, предсказанное философом, остановило мидо-лидийскую войну.')
    print('=======')
    print()
  ok += 1
  for s in sorted_indices:
    print(w[s], s, sep=' ')
    print()
    qw += 1
    if qw == 5:
        break
