import h5py
import numpy as np
import os

def average_data(dataset="", times=10):
    test_acc = get_all_results(dataset, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))

def get_all_results(dataset="", times=10):
    test_acc = []
    for i in range(times):
        file_name = f"{dataset}_{i}"
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc

def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc