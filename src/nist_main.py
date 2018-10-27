from NISTProcessor import DataPreprocessor
from NISTProcessor import NISTProcessor
import time

__author__ = 'Deliang Yang'


CLS_NUM = 62  # total classes


def run():

    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. Time stamp: ' + current_time)

    model_name = 'lstm'
    nist0 = NISTProcessor(model_name)

    function_flag = 0

    if function_flag == 0:
        # generate the data subset.
        dp0 = DataPreprocessor()
        dp0.run()

    elif function_flag == 1:
        # define a network and run.
        nist0.define_netowrk()
        nist0.train_test_process()
    elif function_flag == 2:
        # continue running from the saved model.
        nist0.train_test_process()

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time-start_time) + ' s')


if __name__ == '__main__':
    run()
