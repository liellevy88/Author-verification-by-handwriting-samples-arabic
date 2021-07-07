from neural_network_monkey_algorithm import test_nn


array_of_diff_vectors = []
two_vecs = [0]*2  # the 2 vectors of the 2 files


def neural_network_returned_vec(letters_directories_name):
    vec1, vec2 = test_nn(letters_directories_name[0], letters_directories_name[1])
    two_vecs[0] = vec1
    two_vecs[1] = vec2
    diff_vector = create_diff_vector(two_vecs[0], two_vecs[1])
    array_of_diff_vectors.append(diff_vector)  # just for train


def create_diff_vector(vec_1, vec_2):
    diff_vector = [0] * len(vec_1)
    for i in range(len(vec_1)):
        diff_vector[i] = abs(vec_1[i] - vec_2[i])
    return diff_vector


