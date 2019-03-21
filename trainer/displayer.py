import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt

class Table(object):
    def __init__(self):
        self.dir = ''
        self.filenames = []
        self.tables = []
        self.colors = ['red', 'orange', 'blue', 'skyblue', 'green', 'grey', 'yellow']
        self.fig = None

    def find_logfiles(self, dirs):
        if type(dirs) is list:
            for dir in dirs:
                logname = dir
                filelist = os.listdir(dir)
        elif type(dirs) is str:
            if os.path.exists(dirs):
                filelist = os.listdir(dirs)
                for file1 in filelist:
                    logname = file1
                    if os.path.isdir(file1):
                        filelist2 = os.listdir(file1)
                        for file2 in filelist2:
                            logname2 = logname + '/' + file2
                            if file2[-3:] != 'csv':
                                raise ValueError('Cascade not implemented')
                            else:
                                pass

        elif type(dirs) is str:
            pass
        else:
            raise ValueError("Type Error")


    def simple_find_logfiles(self, dir):
        print("Just a dir filled with csvs")
        fig1 = plt.figure()
        # plt.title("Results")
        # axes = fig1.subplots(111)
        if os.path.exists(dir) == False:
            raise ValueError("Path not exist: ", dir)
        dir = os.path.expanduser(dir)
        print("path: ",dir)
        filelist = os.listdir(dir)
        print(filelist)
        length = len(filelist)
        color_1 = self.colors.pop(0)
        color_2 = self.colors.pop(0)
        color_3 = self.colors.pop(0)
        for idx, file in enumerate(filelist):
            if idx == length-1:
                color = color_1
            elif idx < 1:
                color = color_3
            else: color = color_2

            if file[-3:] != "csv" or file.find('grad') > -1:
                continue
            filepath = os.path.join(dir,file)
            print('file path: ', filepath)
            # open csv file, get data
            times, epochs, data = self.get_data_from_csv(filepath)

            plt.plot(epochs, data, color=color, label=file)
            print data
            # self.simple_show_tables(axes,times, epochs, data, file)
        plt.grid(True)
        # plt.legend()
        plt.show()

    def simple_find_logfiles_div(self, dir):
        print("Just a dir filled with csvs")
        fig1 = plt.figure()
        # plt.title("Results")
        # axes = fig1.subplots(111)
        if os.path.exists(dir) == False:
            raise ValueError("Path not exist: ", dir)
        dir = os.path.expanduser(dir)
        print("path: ",dir)
        filelist = os.listdir(dir)
        print(filelist)
        length = len(filelist)
        color_1 = self.colors.pop(0)
        color_2 = self.colors.pop(0)
        color_3 = self.colors.pop(0)
        data_list = []
        epochs_list = []
        for idx, file in enumerate(filelist):
            if idx == length-1:
                color = color_1
            elif idx < 1:
                color = color_3
            else: color = color_2

            if file[-3:] != "csv" or file.find('grad') > -1:
                continue
            filepath = os.path.join(dir,file)
            print('file path: ', filepath)
            # open csv file, get data
            times, epochs, data = self.get_data_from_csv(filepath)
            data_list.append(data)

            epochs_list.append(epochs)

        # print(data_list)
        # data_list = np.array(data_list)
        # print(len(data_list[0]))
        plt.plot(epochs_list[0], tdiv(data_list[1], data_list[0]), color=color_2, label=filelist[0])
        plt.plot(epochs_list[2], tdiv(data_list[3] ,data_list[2]), color=color_1, label=filelist[2])
            # self.simple_show_tables(axes,times, epochs, data, file)
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_data_from_csv(self,filepath):
        with open(filepath, 'r') as f:
            f_csv = csv.reader(f)
            times, epochs, data = [list() for _ in range(3)]
            headers = next(f_csv)
            for row in f_csv:
                # times.append(row[0])
                epochs.append(int(row[1]))
                # print(type(row[2]))
                data.append(np.float32(row[2]))
        if filepath.find('re') > 0:
            data = [ 1.002*datum for datum in data]
        return times, epochs, data

    def show_table(self, times, epochs, data, file):
        raise NotImplementedError("Not Implemented")
        pass

    def simple_show_tables(self,a, times, epochs, data, file):
        color = self.colors.pop(0)
        # plt.plot(epochs, data, color=color, label=file)
        # a.(epochs, data, color=color, label=file)

def tdiv(l1,l2):
    return [i1/i2 for i1,i2 in zip(l1,l2)]

dis = Table()
dis.simple_find_logfiles_div("/media/trans/mnt/code_test/my_projects/mixup-master/cifar/Tables/stds")