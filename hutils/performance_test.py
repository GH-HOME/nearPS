import time
from functools import wraps
import logging

def timeEval(func):
    '''
    Decorator that reports the execution time.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(func.__name__, end-start)
        return result
    return wrapper



class timelogger(object):
    def __init__(self, loggingfile='time.log'):
        self.loggingfile = loggingfile

    def __call__(self, func): # 接受函数
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.info(func.__name__, end - start)
            with open(self.loggingfile, "a") as myfile:
                myfile.write("{} : {} s \n".format(func.__name__, end - start))
                myfile.close()
            return result
        return wrapper


@timelogger(loggingfile='time1.log')
def sub(a, b):
    time.sleep(1)
    return a-b


if __name__ == '__main__':
    sub(22, 24)
