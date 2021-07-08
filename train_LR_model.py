import cv2
from detection_function import start_func
from sklearn.linear_model import LogisticRegression
from monkey_algorithm import array_of_diff_vectors

y_train = []
filename = 'finalized_model.sav'


def prepare_train_lr_model():  # (array_of_diff_vectors_test, y_test):
    for i in range(1, 20):  # 17
        for j in range(i, 20):  # 17
            file1 = cv2.imread('train/page_' + str(i) + "-a.jpeg", 0)  #Image.open('final_solution/page_' + str(i) + "-a.jpeg")
            if i == j:
                file2 = cv2.imread('train/page_' + str(j) + "-b.jpeg", 0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                start_func(file1, file2)
                y_train.append(1)  # same author
            elif i != j and i == 1:
                file2 = cv2.imread('train/page_' + str(j) + "-a.jpeg", 0)  #Image.open('final_solution/page_' + str(j) + "-a.jpeg")
                start_func(file1, file2)
                y_train.append(0)  # different author
                # for k in range(2):
                #     if k == 0:  # and i != j:
                #         file2 = cv2.imread('train/page_' + str(j) + "-a.jpeg", 0)  #Image.open('final_solution/page_' + str(j) + "-a.jpeg")
                #         start_func(file1, file2)
                #         y_train.append(0)  # different author
                #     else:
                #         file2 = cv2.imread('train/page_' + str(j) + "-b.jpeg", 0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                #         start_func(file1, file2)
                #         y_train.append(0)  # different author

    train_lr_model()  # (array_of_diff_vectors_test, y_test)


def train_lr_model():  # (array_of_diff_vectors_test, y_test):
    logreg = LogisticRegression(penalty='l2', C=1, max_iter=len(array_of_diff_vectors[0]), class_weight='balanced')
    logreg.fit(array_of_diff_vectors, y_train)

    array_of_diff_vectors.clear()
    y_test = []

    # test
    # file1 = cv2.imread('test/page_' + str(3) + "-a.jpeg", 0)  # Image.open('final_solution/page_' + str(i) + "-a.jpeg")
    # file2 = cv2.imread('test/page_' + str(3) + "-b.jpeg", 0)  # Image.open('final_solution/page_' + str(i) + "-a.jpeg")
    # counter = 0
    # start_func(file1, file2)
    # y_test.append(1)  # same author

    for i in range(1, 11):
        for j in range(i, 11):
            file1 = cv2.imread('test/page_' + str(i) + "-a.jpeg",
                               0)  # Image.open('final_solution/page_' + str(i) + "-a.jpeg")
            if i == j:
                file2 = cv2.imread('test/page_' + str(j) + "-b.jpeg",
                                   0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                start_func(file1, file2)
                y_test.append(1)  # same author
            elif i != j and i == 1:
                file2 = cv2.imread('test/page_' + str(j) + "-a.jpeg", 0)  # Image.open('final_solution/page_' + str(j) + "-a.jpeg")
                start_func(file1, file2)
                y_test.append(0)  # different author
                for k in range(2):
                    if k == 0:  # and i != j:
                        file2 = cv2.imread('test/page_' + str(j) + "-a.jpeg",
                                           0)  # Image.open('final_solution/page_' + str(j) + "-a.jpeg")
                        start_func(file1, file2)
                        y_test.append(0)  # different author
                    else:
                        file2 = cv2.imread('test/page_' + str(j) + "-b.jpeg",  # final_test_files
                                           0)  # Image.open('final_solution/page_' + str(j) + "-b.jpeg")
                        start_func(file1, file2)
                        y_test.append(0)  # different author

    result = logreg.score(array_of_diff_vectors, y_test)
    predictions = logreg.predict(array_of_diff_vectors)
    predict = logreg.predict_proba(array_of_diff_vectors)

    print("accuracy:")
    print(result)
    print("predict_proba:")
    print(predict)
    # cm = metrics.confusion_matrix(y_test, predictions)
    # heatmap = plt.axes()
    # heatmap = sn.heatmap(cm, annot=True, fmt='g', cmap="Blues")
    # heatmap.set_title('confusion_matrix')
    # plt.savefig("test.png")
    # pickle.dump(logreg, open(filename, 'wb'))


# if __name__ == "__main__":
#     prepare_train_lr_model()

