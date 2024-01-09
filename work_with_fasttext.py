import requests
import pymorphy2 as pm
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import defaultdict, Counter
from razdel import sentenize
import fasttext


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
