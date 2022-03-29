from multiprocessing import Process, Manager
from random import randrange

def f():
    for _ in range(99999):
        print(d[randrange(0,25)])

if __name__ == '__main__':
    # Init
    #manager = Manager()
    #d = manager.dict()
    # Copy from pickle dict to manager dict
    #d[1] = '1'
    #d['2'] = 2

    d = {i:str(i) for i in range(100)}

    workers = 28
    processes = []
    for worker in range(workers):
        p = Process(target=f)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()