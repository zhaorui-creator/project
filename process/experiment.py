import shapefile # 这个
import shapely.geometry as geometry
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib
import numpy as np
import pylab as plt

if __name__ == '__main__':
    # 取得深圳的shape
    sz_shp = shapefile.Reader(
        r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_CHN_shp\gadm36_CHN_2')
    for city_rcd in sz_shp.shapeRecords():  # 遍历每一条shaperecord
        if city_rcd.record[6] == 'Shenzhen':  # 遍历时，record字段是地区的信息（由字符串表示）
            sz_shp = city_rcd.shape  # 遍历时，shape字段是shape——形状（由点组成）

    m = Basemap(llcrnrlon=113.7, llcrnrlat=22.35, urcrnrlon=114.7, urcrnrlat=22.9,
                rsphere=(6378137.00, 6356752.3142),
                resolution='l', area_thresh=1000., projection='lcc', lat_1=22.5, lat_0=22.5, lon_0=114)  # - shape文件，画出区域轮廓
    # 生成一个矩形点阵
    linear_lon = np.linspace(113.77, 114.59, 50)  # 经向50个点
    linear_lat = np.linspace(22.47, 22.83, 25)  # 纬向25个点
    grid_lon, grid_lat = np.meshgrid(linear_lon, linear_lat)  # 构成了一个坐标矩阵，实际上也就是一个网格，两者是同型矩阵
    flat_lon = grid_lon.flatten()  # 将坐标展成一维
    flat_lat = grid_lat.flatten()

    m.readshapefile(r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_HKG_shp\gadm36_HKG_0', 'states', color='grey') #HongKong
    m.readshapefile(r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_CHN_shp\gadm36_CHN_2', 'states', color='grey') #Mainland in given lon and lat
    m.readshapefile(r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_CHN_shp\gadm36_CHN_2', 'Shenzhen', color='red')
    m.scatter(flat_lon, flat_lat, latlon=True, s=60, marker="o") # latlon这个参数指明我传入的数据是经纬度，默认是像素点坐标
    m.scatter(114, 22.5, latlon=True, s=60, marker="o") # 这是一个测试点
    m.drawmeridians(np.arange(10, 125, 0.5), labels=[1, 0, 0, 1]) # 画经线
    m.drawparallels(np.arange(15, 30, 0.3),labels=[1,0,0,0])  #画纬度平行线
    plt.show()
    plt.clf()

    # np.column_stack((a,b)):向矩阵a增加列，b是增加的部分，将1维数组转换成2维，这样flat的每个点对应上面xi,yi的所有点
    flat_points = np.column_stack((flat_lon, flat_lat))

    in_shape_points = []
    for pt in flat_points:
        # make a point and see if it's in the polygon
        if geometry.Point(pt).within(geometry.shape(sz_shp)):
            in_shape_points.append(pt)
            print("The point is in SZ")
        else:
            print("The point is not in SZ")
    selected_lon = [elem[0] for elem in in_shape_points]
    selected_lat = [elem[1] for elem in in_shape_points]

    m.readshapefile(r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_HKG_shp\gadm36_HKG_0', 'states', color='grey') #HongKong
    m.readshapefile(r'G:\Shenzhen-wind\深圳地形文件\shapefile\gadm36_CHN_shp\gadm36_CHN_2', 'states', color='grey') #Mainland in given lon and lat
    m.scatter(selected_lon, selected_lat, latlon=True, s=60, marker="o")
    m.scatter(114, 22.5, latlon=True, s=60, marker="o") # 这是一个测试点
    m.drawmeridians(np.arange(10, 125, 0.5), labels=[1, 0, 0, 1])
    m.drawparallels(np.arange(15, 30, 0.3),labels=[1,0,0,0])  #画纬度平行线

    plt.show()
