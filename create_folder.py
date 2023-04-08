import os
import numpy
from random import shuffle

PATH = '../ISIC2018/ISIC2018_Task1-2_Training_Input'
SAVE_PATH = './Datasets'
def create_5_folder(folder, save_foler):
    file_list = os.listdir(folder)
    shuffle(file_list)

    FOLDER_SIZE = len(file_list) // 5
    SPLIT_SIZE = FOLDER_SIZE // 2 # size of the validation or training chunk, validation is half the folder size


    for i in range(5):
        if i != 0: # folders 1-4
            pre_test_list = file_list[0:i*FOLDER_SIZE]
        else:
            pre_test_list = []
        test_list = file_list[i*FOLDER_SIZE:(i+1)*FOLDER_SIZE]

        if i < 4:
            start_ix = (i+1)*FOLDER_SIZE
            valid_list = file_list[start_ix: start_ix + SPLIT_SIZE]
            train_list = file_list[start_ix + SPLIT_SIZE:] + pre_test_list
        else:
            valid_list = file_list[:SPLIT_SIZE]
            train_list = file_list[SPLIT_SIZE:i*FOLDER_SIZE]

        if not os.path.isdir(save_foler + '/folder'+str(i+1)):
            os.makedirs(save_foler + '/folder'+str(i+1))

        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_train.list'), train_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_validation.list'), valid_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_test.list'), test_list)


def text_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


if __name__ == "__main__":
    create_5_folder(PATH, SAVE_PATH)
