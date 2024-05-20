
# Report 

## Scripts

- `train.py` - Training script
  - **Inputs**: Training data, (optional) hyperparameter values
  - **Outputs**: Serialized model file
- `eval.py` - Evaluation script
  - **Inputs**: Serialized model file, evaluation data (in the JSONL format defined above)
  - **Outputs**: Accuracy, optionally confusion matrix and precision/recall numbers for each category
- `classify.py` - Classification script
  - **Inputs**: Serialized model file, data to classify (JSONL without `category` field)
  - **Outputs**: The input JSONL with `category` field set to the model's predicted value
- `dataAnalysis.py` - Script to get a basic overview of the dataset
- `NN_model.py` - PyTorch model using pretrained embeddings from GloVe
- `GLOBAL_PARAMETERS.py` - Script to keep model parameters in one place

## Data Analysis 

- Dataset is small
- No headline has more than 44 words
- 99% of short descriptions fit into 60 words
- Dataset includes date, author, and link
- Dataset is unbalanced
- If everything were classified as the most frequent class, the final accuracy would be 16.9%

### Decisions Made According to Analysis 

- Decided to work only with headline and short description
- Did not use authors' names because there were 3541 unique names out of 10000, so the names were not repeated enough to derive any relationships
- I used date for extracting the day of the week for a while to test if there was any connection between the day of publishing an article and its type, but this didn't seem to work
- For embeddings, take only the first 60 words from the `short description` and the first 44 from the `headline`

## Data Preprocessing

- Since there are not many words in the dataset, all words were transformed to lowercase 

## Finding the Model 

- According to instruction, accuracy will be our main metric, but since the dataset is unbalanced, it doesn't tell us much about performance on less frequent classes
- At the beginning, tried basic models to get a baseline
  - 16.9% accuracy - classifier predicting the most frequent class
  - Data was transformed using a tf-idf transformer 
  - Tried methods: Logistic Regression, Random Forests, KNN, Gaussian Classifier
  - Best model scored around 42%
- Decided to use pretrained embeddings due to the small dataset
  - To provide some information about the meaning of the words and to consider similar words in a similar way
  - Used embeddings from GloVe model
    - Easy to access, some of the embeddings are trained on twitter contributions (maybe simmilar format as articless)
- Tried different numbers and sizes of hidden layers
- Used dropout and regularization to prevent overfitting 
- Best model accuracy exceeded 47%

## Final Model 
- to train the model embeddengs have to be downloaded an unziped
  - download at: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
  - path to them can be specified in parameters of `train.py` default is embedders/glove.6B.50d.txt

- Final model structure is:
  - Embedding layer with pretrained embeddings(default dimension of one embedding is 50)
  - Dropout layer with dropout = 0.5
  - Classification layer with softmax() function

### Observations During Training 

- Hidden layers didn't help
- Using only dropout and not L2-regularization led to overfitting on training data, but performance on test data kept improving 
- Training the embeddings seemed to help 
- bigger embeddings didn't help (from the same model)

## Potential Improvements 

- Try different embeddings such as word2vec trained on GoogleNews (bigger model and trained on articles)
- Get more data - whole articles, not just summaries
- If possible, find more training examples 
- Create a vocabulary with classes of synonyms and then substitute all synonyms for one representative from the group, shrinking the vocabulary and potentially improving model training
  - Similar idea: substitute each word for its clause instead of synonyms