import random
import itertools
from functools import reduce
import operator
import math

random_list=list(itertools.product(range(0,500),range(0,500)))
#print(random_list)
pointnum_list=[4,5,6,7]
point_num=random.choice(pointnum_list)
coords = random.sample(random_list,point_num)
center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
org_point = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
org_point=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
org_point.reverse()
center_list=[]
for i in org_point:
    i=list(i)
    center_list.append(i)
print(center_list)

