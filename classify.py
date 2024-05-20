
from eval import Evaluator
from train import Dataset
import GLOBAL_PARAMETERS
import copy
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/test.jsonl", type=str, help="Data path")
parser.add_argument("--model_path", default=GLOBAL_PARAMETERS.MODEL_PATH, type=str, help="Model path")
parser.add_argument("--result_path", default="results/test.jsonl", type=str, help="Result file path")
parser.add_argument("--emb_dimension", default=GLOBAL_PARAMETERS.EMB_DIMENSION, type=int, help="length of one embedding: 50 , 100, 200, 300")
parser.add_argument("--use_columns", default= GLOBAL_PARAMETERS.USE_COLUMNS , type=list, help="possible values: ['headline','short_description']")
parser.add_argument("--max_tokens", default= GLOBAL_PARAMETERS.MAX_TOKENS, type=list, help=" [44, 60] ... none of the headlines will be truncated, 0.85 % of short description will be trucated a litle bit ")

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

    