from functools import reduce
import operator
import math
#coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
coords = [(900,1000),(700,800),(500,600),(300,400),(100,200)]
center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
a=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
a.reverse()
new=[]
for i in a:
    i=list(i)
    new.append(i)
print(new)



m=[]
c=[(1,2),(3,4),(5,6)]
for i in c:
    i = list(i)
    print(i)
    m.append(i)
print(m)






