from time import time as now 

def main():
    l = list()
    for i in range(100_000):
        l.append(str(i))
    start = now()
    s = ",".join(l)
    end = now()
    print('Len: ', len(s), 'Time: ', (end - start), 'seconds')

if __name__ == '__main__':
    main()
