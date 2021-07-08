from __future__ import division, print_function, absolute_import
import sys
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torchvision
from PIL import Image, ImageOps
import PIL.ImageOps
import torch.nn.functional as F
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from networkx.drawing.tests.test_pylab import plt
import seaborn as sn  # heatmap
from resizeimage import resizeimage
from tqdm import tqdm
import numpy as np
import torch
from itertools import chain  # flatten list


def sort(path):
    return int(path.split('\\')[1].split('.')[0])


def sort_our_dataset(path):
    return int(path.split('_')[1].split('.')[0])


def confusion_matrix(y_pred, y_real, DT_predict_prob):
    y_real = y_real.astype(int)
    array_different_letters = []
    array_different_letters.append(y_pred[0])
    for i in range(1, len(y_pred)):
        if y_pred[i] not in array_different_letters:
            array_different_letters.append(int(y_pred[i]))

    for i in range(len(y_real)):
        if y_real[i] not in array_different_letters:
            array_different_letters.append(int(y_real[i]))

    c_m_len = np.max(array_different_letters) + 1
    confusion_mat = np.zeros((c_m_len, c_m_len))
    for i in range(len(y_real)):
        if max(DT_predict_prob[i]) > 0.6:
            confusion_mat[y_real[i]][y_pred[i]] += 1

    counter_right = 0
    counter_all = 0
    for i in range(c_m_len):
        for j in range(c_m_len):
            if i != j:
                counter_all += confusion_mat[i][j]
            else:
                counter_right += confusion_mat[i][j]
                counter_all += confusion_mat[i][j]

    accuracy = counter_right / counter_all
    # print("accuracy after:" + str(accuracy))
    # heatmap = plt.axes()
    # heatmap = sn.heatmap(confusion_mat, annot=True, fmt='g', cmap="Blues")
    # heatmap.set_title('confusion_matrix')
    # plt.savefig("result.png")


def load_dataset():
    trainx = []
    path_train = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/allData")):
        path = os.path.join("C:/Users/aviga/Desktop/allData", img)
        path_train.append(path)

    sort_path_train = sorted(path_train, key=sort)
    for path in sort_path_train:
        img = Image.open(path)
        if img.size[0] < 32 and img.size[1] < 32:
            continue
        img = resizeimage.resize_cover(img, [32, 32])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0],
                std=[1],
            ),
        ])
        normalized_img = transform(img)

        img = np.asarray(normalized_img)
        trainx.append(img[0])

    trainx = np.asarray(trainx)
    trainx = trainx.reshape([-1, 1, 32, 32])
    tensor_x = torch.Tensor(trainx)

    # train_y
    trainy = pd.read_csv("C:/Users/aviga/Desktop/y_allData.csv", header=None)
    trainy = trainy.values.astype('int32') - 1
    tensor_y = torch.Tensor(trainy)

    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        num_workers=0,
        shuffle=False
    )

    return train_loader


def load_test_dataset(directory_name):
    # test
    testx = []
    path_test = []

    for img in tqdm(os.listdir(directory_name)):
        path = os.path.join(directory_name, img)
        path_test.append(path)

    sort_path_test_2 = sorted(path_test, key=sort_our_dataset)
    for path in sort_path_test_2:
        img = Image.open(path)
        if img.size[0] < 32 and img.size[1] < 32:
            continue
        img = resizeimage.resize_cover(img, [32, 32])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0],
                std=[1],
            ),
        ])
        normalized_img = transform(img)

        img = np.asarray(normalized_img)
        testx.append(img[0])

    testx = np.asarray(testx)

    testx = testx.reshape([-1, 1, 32, 32])
    tensor_x = torch.Tensor(testx)

    #  testing data
    test_loader = torch.utils.data.DataLoader(
        tensor_x,
        batch_size=100,
        num_workers=0,
        shuffle=False
    )

    return test_loader, sort_path_test_2


class Net(nn.Module):
    # Defining the Constructor
    def __init__(self, num_classes=30):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        prob = F.log_softmax(x, dim=1)
        return prob


def train(model, device, train_loader, optimizer, epoch, loss_criteria):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the optimizer
        # Push the data forward through the model layers
        output = model(data)
        target = target.detach().numpy()
        target = target.flatten()
        final_target = torch.tensor(target)
        final_target = final_target.type(torch.LongTensor)
        loss = loss_criteria(output, final_target)
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader, loss_criteria):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    output_classes_arr = []
    with torch.no_grad():
        batch_count = 0
        for data in test_loader:
            batch_count += 1
            data = data.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            sm = torch.nn.Softmax()
            probabilities = sm(output)
            probabilities = probabilities.detach().numpy()
            output_np = output.detach().numpy()
            output_classes = np.argmax(output_np, axis=1)
            output_classes_arr.append(output_classes)

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    output_classes_arr = list(chain.from_iterable(output_classes_arr))
    return output_classes_arr, probabilities


def train_neural_network():
    train_loader = load_dataset()
    device = "cpu"
    # if (torch.cuda.is_available()):
    #     device = "cuda"
    classes = 30
    model = Net().to(device)
    loss_criteria = nn.CrossEntropyLoss()
    # Train over 10 epochs (We restrict to 10 for time issues)
    Network = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # device = "cuda"
    epochs = 5
    epoch_nums = []
    training_loss = []
    for epoch in range(1, epochs + 1):
        if epoch == 5:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        train_loss = train(model, device, train_loader, optimizer, epoch, loss_criteria)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)

    # confusion_matrix(y_pred, y_real, DT_predict_prob)
    # graph loss
    # plt.figure(figsize=(15, 15))
    # plt.plot(epoch_nums, training_loss)
    # plt.plot(epoch_nums, validation_loss)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train', 'test'], loc='upper right')
    # # plt.savefig("filname.png")
    # plt.show()


def save_checkpoint(state, file_name="my_checkpoint_auto.pth.tar"):
    torch.save(state, file_name)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])


def test_nn_auto(letters_directory_name_1, letters_directory_name_2):
    # Settings
    test_loader_f1, sort_path_test_f1 = load_test_dataset(letters_directory_name_1)
    test_loader_f2, sort_path_test_f2 = load_test_dataset(letters_directory_name_2)
    loss_criteria = nn.CrossEntropyLoss()
    device = "cpu"
    # if (torch.cuda.is_available()):
    #     device = "cuda"
    model_l = Net().to(device)

    load_checkpoint(torch.load("my_checkpoint_auto.pth.tar"), model_l)
    y_predict_f1, prob_f1 = test(model_l, device, test_loader_f1, loss_criteria)
    y_predict_f2, prob_f2 = test(model_l, device, test_loader_f2, loss_criteria)

    paths_mim = []
    paths_nun = []
    paths_wow = []
    paths_lamAlif = []
    paths_mimAlif = []

    for i in range(len(prob_f1)):
        if max(prob_f1[i]) > 0.4:
            if y_predict_f1[i] == 23:
                paths_mim.append(sort_path_test_f1[i])
            elif y_predict_f1[i] == 24:
                paths_nun.append(sort_path_test_f1[i])
            elif y_predict_f1[i] == 26:
                paths_wow.append(sort_path_test_f1[i])
            elif y_predict_f1[i] == 28:
                paths_lamAlif.append(sort_path_test_f1[i])
            elif y_predict_f1[i] == 29:
                paths_mimAlif.append(sort_path_test_f1[i])


    paths_mim_f2 = []
    paths_nun_f2 = []
    paths_wow_f2 = []
    paths_lamAlif_f2 = []
    paths_mimAlif_f2 = []
    paths_after_threshold_f2 = []
    for i in range(len(prob_f2)):
        if max(prob_f2[i]) > 0.4:
            if y_predict_f2[i] == 23:
                paths_mim_f2.append(sort_path_test_f2[i])
            elif y_predict_f2[i] == 24:
                paths_nun_f2.append(sort_path_test_f2[i])
            elif y_predict_f2[i] == 26:
                paths_wow_f2.append(sort_path_test_f2[i])
            elif y_predict_f2[i] == 28:
                paths_lamAlif_f2.append(sort_path_test_f2[i])
            elif y_predict_f2[i] == 29:
                paths_mimAlif_f2.append(sort_path_test_f2[i])

    return paths_mim, paths_nun, paths_wow, paths_lamAlif, paths_mimAlif, \
           paths_mim_f2, paths_nun_f2, paths_wow_f2, paths_lamAlif_f2, paths_mimAlif_f2




    # total_sum_predict_classes_f1 = np.zeros(30)
    # total_sum_predict_classes_f2 = np.zeros(30)
    #
    # for i in range(len(y_predict_f1)):
    #     total_sum_predict_classes_f1[y_predict_f1[i]] += 1
    #
    # for i in range(len(y_predict_f2)):
    #     total_sum_predict_classes_f2[y_predict_f2[i]] += 1
    #
    # print("file 1")
    # print(total_sum_predict_classes_f1)
    # print("file 2")
    # print(total_sum_predict_classes_f2)
    #
    # return total_sum_predict_classes_f1.astype(int), total_sum_predict_classes_f2.astype(int)


# if __name__ == "__main__":
    # train_neural_network()
    # test_nn()


