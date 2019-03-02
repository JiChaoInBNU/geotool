from PyQt5.QtCore import *
import threading

log_path = "src/log/log.txt"
def reset_log():
    with open(log_path, 'w') as w:
        w.write(str(0))

class ThreadWrapper(threading.Thread):
    def __init__(self, func, *args):
        threading.Thread.__init__(self)
        self.work = True
        self.args = args
        self.func = func
    def run(self):
        reset_log()
        self.func(self.args)

class ProcessHandler(QThread):
    process_signal = pyqtSignal(int)
    def __init__(self, parent=None):
        super(ProcessHandler, self).__init__(parent)

        self.work = True

    def run(self):
        while True:
            with open(log_path, 'r') as r:
                v = r.readline()
                if v == '':
                    continue
                else:
                    self.process_signal.emit(int(v))
                    print(v)
                    if int(v) >= 100:
                        reset_log()
                        self.__del__()
                        break

    def __del__(self):
        self.work = False
        self.wait()