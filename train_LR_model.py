import cv2
from PIL import Image
from detection_function import start_func
from sklearn.linear_model import LogisticRegression
import pickle
from monkey_algorithm import array_of_diff_vectors

y_train = []
filename = 'finalized_model.sav'


def prepare_train_lr_model(array_of_diff_vectors_test, y_test):
    for i in range(1, 17):  # 17
        for j in range(i, 17):  # 17
            file1 = cv2.imread('final_solution/page_' + str(i) + "-a.jpeg", 0)  #Image.open('final_solution/page_' + str(i) + "-a.jpeg")
            if i == j:
                file2 = cv2.imread('final_solution/page_' + str(j) + "-b.jpeg", 0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                start_func(file1, file2)
                y_train.append(1)  # same author
            elif i != j:
                for k in range(2):
                    if k == 0:  # and i != j:
                        file2 = cv2.imread('final_solution/page_' + str(j) + "-a.jpeg", 0)  #Image.open('final_solution/page_' + str(j) + "-a.jpeg")
                        start_func(file1, file2)
                        y_train.append(0)  # different author
                    else:
                        file2 = cv2.imread('final_solution/page_' + str(j) + "-b.jpeg", 0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                        start_func(file1, file2)
                        y_train.append(0)  # different author


    train_lr_model(array_of_diff_vectors_test, y_test)


def train_lr_model(array_of_diff_vectors_test, y_test):
    logreg = LogisticRegression(penalty='l2', C=1, max_iter=len(array_of_diff_vectors[0]))
    logreg.fit(array_of_diff_vectors, y_train)
    result = logreg.score(array_of_diff_vectors_test, y_test)
    print("accuracy:")
    print(result)
    # pickle.dump(logreg, open(filename, 'wb'))


# if __name__ == "__main__":
#     prepare_train_lr_model()

