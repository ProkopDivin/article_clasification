
import argparse 
from eval import Evaluator
from train import Dataset
import numpy
import copy
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/test.jsonl", type=str, help="Data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")
parser.add_argument("--result_path", default="results/test.jsonl", type=str, help="Result file path")

class Classifier:

    def save_results(result_path, dataframe):
        json_data = dataframe.to_json(orient='records', lines=True)
        with open(result_path, 'w') as f:
            f.write(json_data)

    def classify(data_path, model_path, result_path):      
        data = Dataset(data_path)
        original_data =copy.deepcopy(data.data)
        evaluator = Evaluator(model_path)
        x, _ = evaluator.prepare_data(data)
        pred_y = evaluator.predict(x)      
        labels = [[Dataset.classes[i]] for i in pred_y]
        original_data['category'] = labels
        Classifier.save_results(result_path, original_data)




if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    Classifier.classify(args.data_path, args.model_path, args.result_path)

    