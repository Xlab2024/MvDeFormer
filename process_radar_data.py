import os, glob
import pdb
from sklearn.model_selection import train_test_split
seed = 42
def generate_filelist(data_dir, suffix='*.jpg'):
    trainlist = []
    testlist = []
    validlist = []
    labellist_train = []
    labellist_test = []
    labellist_valid = []

    for user in os.listdir(data_dir):
        train_valid_subjects, test_subjects = train_test_split(os.listdir(os.path.join(data_dir, user)), test_size=0.2,
                                                         random_state=seed)
        train_subjects, valid_subjects = train_test_split(train_valid_subjects, test_size=0.125,
                                                         random_state=seed)
        for action_option in train_subjects:
            labellist_train.append(action_option.split('-')[0])
            trainlist.append(glob.glob(os.path.join(data_dir, user, action_option, suffix)))
        for action_option in test_subjects:
            labellist_test.append(action_option.split('-')[0])
            testlist.append(glob.glob(os.path.join(data_dir, user, action_option, suffix)))
        for action_option in valid_subjects:
            labellist_valid.append(action_option.split('-')[0])
            validlist.append(glob.glob(os.path.join(data_dir, user, action_option, suffix)))
    return trainlist  ,testlist , validlist , labellist_train, labellist_test,labellist_valid


def generate_image_files(data_dir, output_file):
    train_list,test_list,valid_list, label_list_train, label_list_test,label_list_valid = generate_filelist(data_dir)
    with open(output_file, 'w') as f:
        for file_name, label in zip( train_list, label_list_train):
            file_name.sort()
            file_name.append(label)
            f.write(','.join(file_name))
            f.write('\n')
    with open(output_file1, 'w') as f:
        for file_name, label in zip(test_list, label_list_test):
            file_name.sort()
            file_name.append(label)
            f.write(','.join(file_name))
            f.write('\n')
    with open(output_file2, 'w') as f:
        for file_name, label in zip(valid_list, label_list_valid):
            file_name.sort()
            file_name.append(label)
            f.write(','.join(file_name))
            f.write('\n')


if __name__ == '__main__':
    data_dir = './data_picture'
    suffix = '*.jpg'
    output_file = 'radar_data_trainlist.txt'
    output_file1 = 'radar_data_testlist.txt'
    output_file2 = 'radar_data_validlist.txt'
    generate_image_files(data_dir, output_file)

