import os
from detection_function_auto_encoder import start_func_auto
import numpy as np
from neural_network_auto_encoder import test_nn_auto, train_neural_network
import cv2
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from resizeimage import resizeimage
from PIL import Image, ImageOps


def sort(path):
    return int(path.split('_')[1].split('.')[0])


def prepare_image(image):
    image = resizeimage.resize_cover(image, [32, 32])
    image = np.asarray(image)/255
    image = image.reshape([-1, 1, 32, 32])
    image = torch.Tensor(image)
    return image


class Autoencoder_train(nn.Module):
    def __init__(self):
        super(Autoencoder_train, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 12),
            nn.ReLU(True),
            nn.Linear(12, 1)
                                     )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 3),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.encoder(x)
        encoded_x = x
        # x = self.decoder(x)
        return x, encoded_x


class Autoencoder_test(nn.Module):
    def __init__(self):
        super(Autoencoder_test, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 3),
                                     nn.ReLU()
                                     )

    def forward(self, x):
        x = self.encoder(x)
        return x


def create_diff_vector(vec_1, vec_2):
    diff_vector = [0] * len(vec_1)
    for i in range(len(vec_1)):
        diff_vector[i] = abs(vec_1[i] - vec_2[i])
    return diff_vector


if __name__ == "__main__":
    model = Autoencoder_train()

    # for train auto encoder - same author
    train_auto_encoder = []
    id_of_author = []
    root_dir = "C:/Users/aviga/Desktop/train_auto_encoder"
    files_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for file in files_dirs:
        letter_dirs = [os.path.join(file, f) for f in os.listdir(file)]
        for letter_folder in letter_dirs:
            path_train = []
            for img in tqdm(os.listdir(letter_folder)):
                path = os.path.join(letter_folder, img)
                path_train.append(path)
            for i in range(len(path_train)):
                for j in range(i, len(path_train)):
                    letter1 = Image.open(path_train[i])
                    letter1 = prepare_image(letter1)
                    recon1, encoded_img1 = model(letter1)
                    encoded_img1 = encoded_img1.detach().numpy()
                    encoded_img1 = encoded_img1.flatten()
                    letter2 = Image.open(path_train[j])
                    letter2 = prepare_image(letter2)
                    recon2, encoded_img2 = model(letter2)
                    encoded_img2 = encoded_img2.detach().numpy()
                    encoded_img2 = encoded_img2.flatten()
                    diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                    train_auto_encoder.append(diff_vector)
                    id_of_author.append(1)  # same author

    # for train auto encoder - different author
    root_dir = "C:/Users/aviga/Desktop/train_auto_encoder_different_author"
    files_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for file in files_dirs:
        for img in tqdm(os.listdir(letter_folder)):
            path = os.path.join(letter_folder, img)
            path_train.append(path)
        sort_path_train = sorted(path_train, key=sort)
        letter1 = Image.open(sort_path_train[0])
        letter1 = prepare_image(letter1)
        recon1, encoded_img1 = model(letter1)
        encoded_img1 = encoded_img1.detach().numpy()
        encoded_img1 = encoded_img1.flatten()
        for j in range(1, len(sort_path_train)):
            letter2 = Image.open(sort_path_train[j])
            letter2 = prepare_image(letter2)
            recon2, encoded_img2 = model(letter2)
            encoded_img2 = encoded_img2.detach().numpy()
            encoded_img2 = encoded_img2.flatten()
            diff_vector = create_diff_vector(encoded_img1, encoded_img2)
            train_auto_encoder.append(diff_vector)
            id_of_author.append(0)  # different author

    train_neural_network()

    file1 = cv2.imread('C:/Users/aviga/Desktop/test_autoencoder_same_author/page_39-a.jpeg', 0)
    file2 = cv2.imread('C:/Users/aviga/Desktop/test_autoencoder_same_author/page_39-b.jpeg', 0)
    start_func_auto(file1, file2)
    paths_mim, paths_nun, paths_wow, paths_lamAlif, paths_mimAlif, \
    paths_mim_f2, paths_nun_f2, paths_wow_f2, paths_lamAlif_f2, paths_mimAlif_f2 = test_nn_auto\
        ('lettersFile1', 'lettersFile2')

    test_auto_encoder = []
    if len(paths_mim) != 0 and len(paths_mim_f2) != 0:
        for image_path in paths_mim:
            letter1 = Image.open(image_path)
            if letter1.size[0] < 32 and letter1.size[1] < 32:
                continue
            letter1 = prepare_image(letter1)
            recon1, encoded_img1 = model(letter1)
            encoded_img1 = encoded_img1.detach().numpy()
            encoded_img1 = encoded_img1.flatten()
            for image_path_2 in paths_mim_f2:
                letter2 = Image.open(image_path_2)
                if letter2.size[0] < 32 and letter2.size[1] < 32:
                    continue
                letter2 = prepare_image(letter2)
                recon2, encoded_img2 = model(letter2)
                encoded_img2 = encoded_img2.detach().numpy()
                encoded_img2 = encoded_img2.flatten()
                diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                test_auto_encoder.append(diff_vector)

    if len(paths_nun) != 0 and len(paths_nun_f2) != 0:
        for image_path in paths_nun:
            letter1 = Image.open(image_path)
            if letter1.size[0] < 32 and letter1.size[1] < 32:
                continue
            letter1 = prepare_image(letter1)
            recon1, encoded_img1 = model(letter1)
            encoded_img1 = encoded_img1.detach().numpy()
            encoded_img1 = encoded_img1.flatten()
            for image_path_2 in paths_nun_f2:
                letter2 = Image.open(image_path_2)
                if letter2.size[0] < 32 and letter2.size[1] < 32:
                    continue
                letter2 = prepare_image(letter2)
                recon2, encoded_img2 = model(letter2)
                encoded_img2 = encoded_img2.detach().numpy()
                encoded_img2 = encoded_img2.flatten()
                diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                test_auto_encoder.append(diff_vector)

    if len(paths_wow) != 0 and len(paths_wow_f2) != 0:
        for image_path in paths_wow:
            letter1 = Image.open(image_path)
            if letter1.size[0] < 32 and letter1.size[1] < 32:
                continue
            letter1 = prepare_image(letter1)
            recon1, encoded_img1 = model(letter1)
            encoded_img1 = encoded_img1.detach().numpy()
            encoded_img1 = encoded_img1.flatten()
            for image_path_2 in paths_wow_f2:
                letter2 = Image.open(image_path_2)
                if letter2.size[0] < 32 and letter2.size[1] < 32:
                    continue
                letter2 = prepare_image(letter2)
                recon2, encoded_img2 = model(letter2)
                encoded_img2 = encoded_img2.detach().numpy()
                encoded_img2 = encoded_img2.flatten()
                diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                test_auto_encoder.append(diff_vector)

    if len(paths_lamAlif) != 0 and len(paths_lamAlif_f2) != 0:
        for image_path in paths_lamAlif:
            letter1 = Image.open(image_path)
            if letter1.size[0] < 32 and letter1.size[1] < 32:
                continue
            letter1 = prepare_image(letter1)
            recon1, encoded_img1 = model(letter1)
            encoded_img1 = encoded_img1.detach().numpy()
            encoded_img1 = encoded_img1.flatten()
            for image_path_2 in paths_lamAlif_f2:
                letter2 = Image.open(image_path_2)
                if letter2.size[0] < 32 and letter2.size[1] < 32:
                    continue
                letter2 = prepare_image(letter2)
                recon2, encoded_img2 = model(letter2)
                encoded_img2 = encoded_img2.detach().numpy()
                encoded_img2 = encoded_img2.flatten()
                diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                test_auto_encoder.append(diff_vector)

    if len(paths_mimAlif) != 0 and len(paths_mimAlif_f2) != 0:
        for image_path in paths_mimAlif:
            letter1 = Image.open(image_path)
            if letter1.size[0] < 32 and letter1.size[1] < 32:
                continue
            letter1 = prepare_image(letter1)
            recon1, encoded_img1 = model(letter1)
            encoded_img1 = encoded_img1.detach().numpy()
            encoded_img1 = encoded_img1.flatten()
            for image_path_2 in paths_mimAlif_f2:
                letter2 = Image.open(image_path_2)
                if letter2.size[0] < 32 and letter2.size[1] < 32:
                    continue
                letter2 = prepare_image(letter2)
                recon2, encoded_img2 = model(letter2)
                encoded_img2 = encoded_img2.detach().numpy()
                encoded_img2 = encoded_img2.flatten()
                diff_vector = create_diff_vector(encoded_img1, encoded_img2)
                test_auto_encoder.append(diff_vector)

    if (len(paths_mim) == 0 or len(paths_mim_f2) == 0) and (len(paths_nun) == 0 or \
            len(paths_nun_f2) == 0) and (len(paths_wow) == 0 or len(paths_wow_f2) == 0) and \
            (len(paths_lamAlif) == 0 or len(paths_lamAlif_f2) == 0) and (len(paths_mimAlif) == 0 or \
            len(paths_mimAlif_f2) == 0):
        print("error")

    else:
        # Train LR model
        logreg = LogisticRegression(penalty='l2', C=1, max_iter=len(train_auto_encoder[0]), class_weight='balanced')
        logreg.fit(train_auto_encoder, id_of_author)

        # Test LR model
        predictions = logreg.predict(test_auto_encoder)
        results = []
        t = logreg.predict_proba(test_auto_encoder)
        for i in range(len(t)):
            index = np.argmax(t[i])
            if max(t[i]) > 0.83:
                results.append(index)
            else:
                results.append(np.abs(index-1))

        same = 0
        different = 0
        for i in range(len(results)):
            if results[i] == 1:
                same += 1
            else:
                different += 1
        print("same_author")
        print(same/(same+different))
        print("different_author")
        print(different/(same+different))














    # model = Autoencoder_train()
    # enc_output = model.encoder(image)
    # recon, encoded_img = model(image)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(encoded_img)
    # model = Autoencoder_train()
    # max_epochs = 50
    # outputs_train, encoded_img = train(model, num_epochs=max_epochs)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(encoded_img)

    # for k in range(0, max_epochs, 2):
    #     plt.figure(figsize=(10, 2))
    #     imgs = outputs_train[k][1].detach().numpy()
    #     recon = outputs_train[k][2].detach().numpy()
    #     for i, item in enumerate(imgs):
    #         if i >= 10: break
    #         img = item[0, :, :]
    #         plt.subplot(1, 10, i + 1)
    #         plt.savefig("avigail")
    #
    #         plt.imshow(img)
    #
    #     for i, item in enumerate(recon):
    #         if i >= 10: break
    #         plt.subplot(2, 10, 10 + i + 1)
    #         img = item[0, :, :]
    #         plt.savefig("liel")
    #         plt.imshow(img)
    #         plt.imshow(item[i].reshape(32, 32))
    #
    # model_test = Autoencoder_test()
    # outputs_test_letter1, outputs_test_letter2 = test(model_test)
    #
    # diff_vector = [0] * len(outputs_test_letter1)
    # for i in range(len(outputs_test_letter1)):
    #     diff_vector[i] = abs(outputs_test_letter1[i] - outputs_test_letter2[i])
    #
    # for k in range(0, max_epochs, 5):
    #     plt.figure(figsize=(20, 2))
    #     imgs = outputs_test[k][0].detach().numpy()
    #     recon = outputs_test[k][1].detach().numpy()
    #     for i, item in enumerate(imgs):
    #         if i >= 20: break
    #         plt.subplot(2, 20, i + 1)
    #         plt.imshow(item[0])
    #
    #     for i, item in enumerate(recon):
    #         if i >= 20: break
    #         plt.subplot(2, 20, 20 + i + 1)
    #         plt.imshow(item[0])
