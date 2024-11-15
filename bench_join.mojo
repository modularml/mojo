from time import now

fn main():
    l = List[String]()
    for i in range(100_000):
        l.append(str(i))
    start = now()
    s = ",".join(l)
    #print(s)
    end = now()
    print('Len: ', len(s), 'Time: ', (end - start) / 1_000_000_000, 'seconds')
