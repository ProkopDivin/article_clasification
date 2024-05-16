# AI dev assignment - News classification

Create a model for text classification based on given data. Make your code available on a public Git repo.

## Data

The data is a subset of News Category Dataset available on Kaggle. It is stored in JSON-per-line (JSONL) format. Important fields:

- `category` - this is the value that should be predicted by the model
- `headline` - headline of the article
- `short_description` - short snippet of text

The data is divided into three files:

- `train.jsonl` - training data
- `dev.jsonl` - validation data; hyperparameter tuning is optional for this assignment, manually coming with 'good enough' values is OK.
- `test.jsonl` - test data for final evaluation

## Deliverables

- `train.py` - training script
  - **inputs**: train data, (optional) hyperparameter values
  - **outputs**: serialized model file
- `eval.py` - evaluation script
  - **inputs**: serialized model file, evaluation data (in the JSONL format defined above)
  - **outputs**: accuracy, optionally confusion matrix and precision/recall numbers for each category
- `classify.py` - classification script
  - **inputs**: serialized model file, data to classify (the JSONL without `category` field)
  - **outputs**: the input JSONL with `category` field set to model's predicted value
- `requirements.txt` - list of packages which your scripts depend on in [Pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- `report.md` - your comments about design decisions/tradeoffs you made, observations about the model and data and possible improvements. Bullet points are fine.
- other files at your discretion

## Notes

- you can use any model type and any publicly available Python library you consider appropriate
- the scripts should be able to run on a regular laptop
- the scripts should not make API calls through network
