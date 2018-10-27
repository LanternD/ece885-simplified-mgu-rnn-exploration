from matplotlib import pyplot as plt
import numpy as np
import csv

__author__ = 'Deliang Yang'
RST_PATH = './result_output/'  # result path


def csv_log_reader(data_file_name):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # data_file_name = 'basic_mnist_log_wrong_c'
    print('Processing file: ' + data_file_name)
    with open(RST_PATH + data_file_name + '_v4.log', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)
        for line_buf in spamreader:
            train_loss.append(float(line_buf[2]))
            train_acc.append(float(line_buf[1]))
            test_loss.append(float(line_buf[4]))
            test_acc.append(float(line_buf[3]))
        csvfile.close()
    train_loss = np.asarray(train_loss)
    train_acc = np.asarray(train_acc)
    test_loss = np.asarray(test_loss)

    test_acc = np.asarray(test_acc)
    if True:
        # takes log on data
        train_loss = np.log10(train_loss)
        test_loss = np.log10(test_loss)

    if True:
        # data smoothing
        test_loss = smoothing(test_loss, 0.6)
        test_acc = smoothing(test_acc, 0.6)
    return train_loss, train_acc, test_loss, test_acc


def result_plot():

    dataset = 'nist'

    l_trn_loss, l_trn_acc, l_tst_loss, l_tst_acc = csv_log_reader('lstm_' + dataset + '_log')
    b_trn_loss, b_trn_acc, b_tst_loss, b_tst_acc = csv_log_reader('basic_' + dataset + '_log')
    v_trn_loss, v_trn_acc, v_tst_loss, v_tst_acc = csv_log_reader('variant_' + dataset + '_log')
    v4_trn_loss, v4_trn_acc, v4_tst_loss, v4_tst_acc = csv_log_reader('variant4_' + dataset + '_log')

    xx = np.linspace(0, 99, 100)

    f, axarr = plt.subplots(2, 2, figsize=(12, 9))
    axarr[0, 0].plot(xx, l_trn_loss, '--')
    axarr[0, 0].plot(xx, b_trn_loss)
    axarr[0, 0].plot(xx, v_trn_loss[:100], ':')
    axarr[0, 0].plot(xx, v4_trn_loss, '-.')
    axarr[0, 0].set_xlabel('Epochs', fontsize='large')
    axarr[0, 0].set_ylabel('Train loss', fontsize='large')
    axarr[0, 0].grid(True)

    axarr[0, 1].plot(xx, l_tst_loss, '--')
    axarr[0, 1].plot(xx, b_tst_loss)
    axarr[0, 1].plot(xx, v_tst_loss[:100], ':')
    axarr[0, 1].plot(xx, v4_tst_loss, '-.')
    axarr[0, 1].set_xlabel('Epochs', fontsize='large')
    axarr[0, 1].set_ylabel('Test loss', fontsize='large')
    axarr[0, 1].grid(True)

    axarr[1, 0].plot(xx, l_trn_acc, '--')
    axarr[1, 0].plot(xx, b_trn_acc)
    axarr[1, 0].plot(xx, v_trn_acc[:100], ':')
    axarr[1, 0].plot(xx, v4_trn_acc, '-.')
    axarr[1, 0].set_xlabel('Epochs', fontsize='large')
    axarr[1, 0].set_ylabel('Train accuracy', fontsize='large')
    axarr[1, 0].grid(True)

    axarr[1, 1].plot(xx, l_tst_acc, '--', label='Standard LSTM Model')
    axarr[1, 1].plot(xx, b_tst_acc, label='Standard MGU Model')
    axarr[1, 1].plot(xx, v_tst_acc[:100], ':', label='Variant MGU Model')
    axarr[1, 1].plot(xx, v4_tst_acc[:100], '-.', label='Variant4 MGU Model')
    axarr[1, 1].set_xlabel('Epochs', fontsize='large')
    axarr[1, 1].set_ylabel('Test accuracy', fontsize='large')
    axarr[1, 1].grid(True)

    # plt.legend(loc='upper left', bbox_to_anchor=(0.46, 1.55), borderaxespad=0., fontsize='large')#2.15
    f.subplots_adjust(bottom=0.08, top=0.95, left=0.07, right=0.95)

    plt.savefig('./' + 'comparison_' + dataset + '_log' + '.png')
    plt.savefig('./' + 'comparison_' + dataset + '_log' + '.pdf')
    plt.show()


def smoothing(data_list, smv):
    list_length = len(data_list)
    smooth_value = data_list[0]
    return_list = [smooth_value]
    for i in range(1, list_length):
        smooth_value = smooth_value * smv + data_list[i] * (1-smv)
        return_list.append(smooth_value)
    # print(data_list)
    # print(return_list)
    return return_list

if __name__ == '__main__':
    result_plot()
