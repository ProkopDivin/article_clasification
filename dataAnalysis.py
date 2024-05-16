
import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import copy


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