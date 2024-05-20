
import pandas as pd 
import json
import copy
import sklearn.metrics
from train import BaseDataset


class DataAnalysis:
    
    def categories(dataset: pd.DataFrame):
        category_distribution = dataset['category'].value_counts()
        print("\nDistribution of categories:")
        print(category_distribution)
        unique_categories = dataset['category'].nunique()
        print("\nNumber of unique categories:", unique_categories)
        print('categories: ', category_distribution.keys())
    
    def stats(dataset: pd.DataFrame):
        train_df = copy.deepcopy(dataset)
        train_df['headline_length'] = train_df['headline'].apply(lambda x: len(x.split()))
        train_df['short_description_length'] = train_df['short_description'].apply(lambda x: len(x.split()))
        print()
        print(train_df.describe())
        
    def num_of_words(name, tresholds):
        for treshold in tresholds:
           short_count = (data[name].str.split().apply(len) < treshold).sum()
           print(f"with treshold: {treshold} there is  {short_count/len(data) * 100} % with less words")

if __name__ == '__main__':
    data = []
    file_path = 'data/train.jsonl'
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    data = pd.DataFrame(data)

    DataAnalysis.categories(data)
    DataAnalysis.stats(data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(data.describe(include='all') )
    print(data.head(1))
    DataAnalysis.num_of_words('short_description', [i for i in range(20,250,10)])
    data_y = [BaseDataset.classes.index(x) for x in data['category']]
    print('accuracy when all is POLITICS: ', sklearn.metrics.accuracy_score([BaseDataset.classes.index('POLITICS')] * len(data), data_y))
   