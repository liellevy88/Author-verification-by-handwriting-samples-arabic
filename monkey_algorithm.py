from neural_network_for_monkey_alg import neural_network

array_of_diff_vectors = []
two_vecs = []  # the 2 vectors of the 2 files


def neural_network_returned_vec(letters_directories_name):
    for i in range(2):
        vec = neural_network(letters_directories_name[i], i)
        two_vecs.append(vec)
        # print(two_vecs)
    diff_vector = create_diff_vector(two_vecs[0], two_vecs[1])
    array_of_diff_vectors.append(diff_vector)  # just for train
    # logistic_reg(diff_vector)


def create_diff_vector(vec_1, vec_2):
    diff_vector = [0] * len(vec_1)
    for i in range(len(vec_1)):
        diff_vector[i] = abs(vec_1[i] - vec_2[i])
    return diff_vector


# def logistic_reg(diff_vector):
#     y_pred = logreg.predict(X_val_temp)

