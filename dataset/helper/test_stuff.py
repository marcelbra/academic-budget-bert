from multiprocessing import Process, Manager

def f(d):
    for _ in range(99999):
        print(d[1])

if __name__ == '__main__':

    # Init
    manager = Manager()
    d = manager.dict()

    # Copy from pickle dict to manager dit
    d[1] = '1'
    d['2'] = 2

    workers = 64
    processes = []
    for worker in range(workers):
        p = Process(target=f, args=(d,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()