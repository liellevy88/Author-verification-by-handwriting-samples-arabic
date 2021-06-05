import pickle
from train_LR_model import filename


def test_lr_model(diff_vec, y_test):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(diff_vec, y_test)
    print("result")
    print(int(result))
    return int(result)
