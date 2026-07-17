import multiprocessing
def square(x):
    return x * x

if __name__ == '__main__':
    p = multiprocessing.Pool(2)
    print('starting')
    r = p.map(square, list(range(10)))
    print('result:', r)
    p.close(); p.join()