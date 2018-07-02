import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv('movies_metadata.csv')

f1=pd.Series(data.index,index=data['title']).drop_duplicates()

def content_base(title, cosine_sim):
    idx = f1[title]
    # similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sorting based on the similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
 #scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]
credits=pd.read_csv('credits.csv')
keywords=pd.read_csv('keywords.csv')
data = data.drop([19730, 29503, 35587])
keywords['id']=keywords['id'].astype('int')
credits['id']=credits['id'].astype('int')
data['id'] = data['id'].astype('int')

data=data.merge(credits, on='id')
data=data.merge(keywords, on='id')
#print(data.head(2))
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for features in features:
    data[features]=data[features].apply(literal_eval)

def d_name(i):
    for x in i:
        if x['job']=='Director':
            return x['name']
        return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []
data['director'] = data['crew'].apply(d_name)

features = ['cast', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(get_list)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    data[feature] = data[feature].apply(clean_data)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
data['soup'] = data.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
metadata =data.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])
print(content_base('The Godfather', cosine_sim2))