import sys, time
sys.path.insert(0, r'J:\PROJECT\AuroraNLP\AuroraNLP')
print('importing...', flush=True)
from AuroraNLP.performance import ProcessPoolExecutor
print('imported', flush=True)

def square(x):
    return x * x

if __name__ == '__main__':
    p = ProcessPoolExecutor(max_workers=2)
    print('starting...', flush=True)
    p.start()
    print('started', flush=True)
    t = time.time()
    result = p.map(square, list(range(10)))
    print('result:', result, 'elapsed:', round(time.time()-t, 2), flush=True)
    p.stop()
    print('done', flush=True)