from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

polygon=Polygon([(60,224),(173,224),(420,497),(3,499),(5,299)])
#geometry.shape(polygon)
print(polygon)

#a=(1,2,3)
#print(a[0])

center_list7=[]
for m in range(0,430,25):
    for n in range(0,450,25):
        center_list7.append((m,n))

        #m=40
        #for n in range(250,400,10):
            #center_list7.append((m,n))

        #for m in range(40,451):
            #for n in range(40,451):
                #center_list6.append((m,n))

        #for i in range(0,len(center_list7)):

            #center = center_list7[i]
        #for i in range(0,1):
            #center=center_list6[i]
    list1=[]
    list2=[]
    for i in range(0,len(center_list7)):
        #print(len(center_list7))
        center=center_list7[i]
        center=Point(center_list7[i][0],center_list7[i][1])
        print(center)

        if center.within(polygon):
            list1.append(center_list7[i])
        else:
            #print(0)
            list2.append(center_list7[i])
    print(len(list1))
    print(len(list2))
    print(len(center_list7))
    


    
        




