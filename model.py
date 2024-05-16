
import pickle
import lzma

class Model:
    def save(model, path):
        with lzma.open(path, "wb") as model_file:
            pickle.dump(model, model_file)

    def load(path):
        with lzma.open(path, "rb") as model_file:
            model = pickle.load(model_file)
        return model

    def prepare_model():
        pass
