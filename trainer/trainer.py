# -*- coding:utf-8 -*-
import yaml
import numpy as np
import sys
import os
import subprocess

if sys.version_info[0] > 2:
    import threading
    import _thread as thread
    import queue
else:
    import thread
    import Queue as queue
import time

# Out to log file
out_log = open('out.log', 'a')
err_log = open('err.log', 'a')
__console__ = sys.stdout
# import torch

"""
TO DO:
1. run complete running for one config
2. 添加程序热更新，自动重启
3. 热更新tasks
4. 同时只能进行N个task （N设置1）
5. 自动恢复功能
"""
_DEBUG = True


def ll(*args, **kwargs):
    # args 为 元祖 tuple
    if _DEBUG:
        # for key, value in iter(kwargs):
        print("[DEBUG]", end=' ')
        # if type(args) is list:
        #     args.insert(0,"[DEBUG] ")
        # elif type(args) is str:
        #     args = '[DEBUG] ' + args
        # else:
        #     raise ValueError('type args: {}'.format(type(args)))
        print(args, **kwargs)


class Trainer(object):
    def __init__(self):
        self.exefile_name = 'test_train.py'
        self.base_dir = os.path.dirname('.')

        self.max_processes = 1
        self.num_running = 0
        self.training = True
        self.waitting_time = 6  # seccond

        self.command = 1
        # 最大任务数为20
        self.task_q = queue.LifoQueue(maxsize=50)


        self.index = 0
        self.loss_detected = np.zeros(20)
        pass


    def check_process(self):
        print('[#] check_process')
        while self.command > 0:
            print('check_process')
            time.sleep(self.waitting_time)
            if self.num_running > 0:
                for idx, process in enumerate(self.process_list):
                    if process.poll() is None:
                        self.process_list.pop(idx)
                        self.num_running -= 1

    def display(self):
        while self.command > 0:
            command = input("Please Type commands: ")
            if type(command) is not int:
                try:
                    command = int(command)
                except:
                    continue
            self.command = command

            if self.command == 0:
                print("Exit. thank u")
                self.tr.join()
                exit(0)
            elif self.command == 1:
                new_task = input("New task(cfg path): ")
                assert type(new_task) is str
                if new_task[-3:] != 'yml':
                    print("Please use COMMAND 2, {}".format(new_task), file=sys.stderr)
                    continue
                new_task = self.wrap_cmd(new_task)
                self.task_q.put(new_task)
            elif self.command == 2:
                new_command = input("New command: ")
                assert type(new_command) is str
                self.task_q.put(new_command)
            elif self.command == 3:
                new_task_list = self.get_tasklist(input('New task list: '))
                if len(new_task_list) > 0:
                    for new_task in new_task_list:
                        self.task_q.put(new_task)
                else:
                    print("The Tasklist is Blank file.", file=sys.stderr)
            elif self.command == 99:
                ll("   task_q_num = {}".format(self.task_q.qsize()))
                ll("process_q_num = {}".format(self.tr.process_q.qsize()))

            if self.tr.is_alive():
                # print("is alive")
                pass
            else:
                print("Restart TaskRunning Thread.")
                self.tr = TaskRunning(task_q=self.task_q)
                self.tr.start()

    def get_tasklist(self, tasklist):
        if not os.path.exists(tasklist):
            # try to correct the path:
            dir = os.path.abspath('.')
            tasklist = os.path.join(dir, tasklist)
            if not os.path.exists(tasklist):
                # raise ValueError("The path of tasklist NOT CORRECT.")
                print("The path of tasklist NOT CORRECT..", file=sys.stderr)
                return list()
        with open(tasklist, 'r') as f:
            lines = f.readlines()
        tasks = list()
        for line in lines:
            line = line.strip('\n')
            if line[:6] != 'python' and line[-4:] == '.yml' and line.find(' ') < 0:
                task = self.wrap_cmd(line)
                tasks.append(task)
            elif line[:6] == 'python':
                tasks.append(line)
            else:
                print("add command: {}".format(line))
                tasks.append(line)
        return tasks

    def wrap_cmd(self, new_task):
        dirname = self.base_dir
        path = os.path.join(dirname, new_task)
        exe = os.path.join(dirname, self.exefile_name)
        return "python {} --cfg={}".format(exe, path)

    def run(self):
        # thread.start_new_thread(self.display, (self,))
        self.tr = TaskRunning(task_q=self.task_q)
        self.tr.start()
        self.display()

    def import_yml(self):
        pass

    def create_yml(self):
        pass

    def detect_loss(self, latest_loss):
        index = self.index
        tmp = 0
        self.loss_detected[self.index % 20] = latest_loss

        if tmp != 0:
            tmp_back = self.loss_detected[index:20]
            tmp_front = self.loss_detected[0:index]
            tmp = np.zeros(20)
            tmp[0:len(tmp_back)] = tmp_back
            tmp[len(tmp_back):20] = tmp_front
        else:
            tmp = self.loss_detected

        # y = tmp[0:10]
        # x = np.arange(10)
        # y_mean = np.mean(y)
        # top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        # bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        # k_1 = top / bottom
        #
        # y = tmp[10:]
        # x = np.arange(10)
        # y_mean = np.mean(y)
        # top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        # bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        # k_2 = top / bottom
        #
        # y = tmp
        # x = np.arange(20)
        # y_mean = np.mean(y)
        # top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        # bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        # k = top / bottom

        ###  update  ###
        self.index += 1

        if k >= 0:
            """  To Stop Training OR Adjust Learning rate  """
            pass
        else:
            if k_2 + k_1 >= 0:
                # Adjust
                pass
            else:
                pass

    def adjust_lr(self):
        """Adjust Learning Rate"""
        pass

    ###########################################################
    ###    Hyper Params Adjusting                           ###
    ###

    def switch_initializer(self):
        pass

    def set_base_lr(self):
        pass

    #####################################################
    ###    Auto Running                               ###

    def run_script(self):
        cfg_file = 'cfgs/' + 'ga_vgg.yml'
        cammand = 'python {} --cfg={}'.format("train.py", )
        try:
            os.system(cammand)
        except:
            pass

    def auto_log(self):
        pass

    def auto_test(self):
        pass

    def auto_restart(self):
        pass

    ######################################################
    ##      Auto Model Modified

    def add_bn(self):
        pass

    def set_activiation(self):
        pass


class TaskRunning(threading.Thread):
    def __init__(self, task_q, max_process=1):
        self.num_process = 0
        self.max_process = max_process
        self.process_list = []
        self.running = 1
        self.task_q = task_q
        self.process_q = queue.LifoQueue(maxsize=max_process)

        super(TaskRunning, self).__init__()

    def run(self):
        print("Start Taskrunning thread.")
        self.running = 1
        while self.running:
            time.sleep(1)
            self.check_process()
            if self.task_q.qsize() > 0:
                if self.process_q.qsize() < self.max_process:
                    print("Start New Process")
                    task_command = self.task_q.get()
                    print("\nExecute: ", task_command)
                    assert type(task_command) is str, 'Crazy'
                    # obj = subprocess.Popen([task_command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    task_command = task_command.split(' ')
                    assert type(task_command) is list, 'Error'
                    obj = subprocess.Popen(task_command, stdout=out_log, stderr=err_log)
                    self.process_q.put(obj)

    def check_process(self):
        length = self.process_q.qsize()
        sys.stdout.flush()
        # ll('Checking process_num={}'.format(length))
        # ll('Checking  tasklist_n={}'.format(self.task_q.qsize()))
        for i in range(length):
            obj = self.process_q.get()
            if obj.poll() == None:
                self.process_q.put(obj)
            else:
                ll("Loose process_q")
                self.num_process -= 1
                obj.terminate()

    def join(self):
        while not self.process_q.empty():
            obj = self.process_q.get()
            obj.terminate()
        self.running = 0
        super(TaskRunning, self).join()
