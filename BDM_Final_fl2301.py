import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkContext
import sys
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import IPython
from pyproj import Transformer
import shapely
from shapely.geometry import Point
from operator import add

if __name__=='__main__':
    sc = pyspark.SparkContext.getOrCreate()
    spark = SparkSession(sc)
    supermarket = set(map(lambda x: x[9], pd.read_csv('nyc_supermarkets.csv').to_numpy()))
    cbg = dict(map(lambda x: (x[0],(x[1],x[2])), pd.read_csv('nyc_cbg_centroids.csv').to_numpy()))

    def outforma(values):
      temp_dict = {'2019-03':None, '2019-10':None, '2020-03':None, '2020-10':None}
      for each in values:
        temp_dict[each[0]]=each[1]
      return (temp_dict['2019-03'],temp_dict['2019-10'],temp_dict['2020-03'],temp_dict['2020-10'])


    def filfilter(partID, part):
      if partID == 0: next(part)
      for x in csv.reader(part):
        if x[0] in supermarket:
          if (x[12].split('-')[0]+'-'+x[12].split('-')[1] in ['2019-03','2019-10','2020-03','2020-10']):
            date = x[12].split('-')[0]+'-'+x[12].split('-')[1]
            if (x[13].split('-')[0]+'-'+x[13].split('-')[1] in ['2019-03','2019-10','2020-03','2020-10']):
              date = x[13].split('-')[0]+'-'+x[13].split('-')[1]
              location_poi = cbg.get(float(x[18]),None)
              if not (x[19].strip('{}') is ''):
                for each in x[19].strip('{}').split(','):
                  temp = float(each.split(':')[0].replace('"',''))
                  location_home = cbg.get(temp,None)
                  if location_poi and location_home:
                    t = Transformer.from_crs(4326, 2263)
                    distance = Point(t.transform(location_poi[0],location_poi[1])).distance(Point(t.transform(location_home[0],location_home[1])))/5280
                    visitor = float(each.split(':')[1].replace('"',''))
                    yield ((x[18],date),(distance*visitor,visitor))
    
    weeklypattern = sc.textFile("/tmp/bdm/weekly-patterns-nyc-2019-2020/*").mapPartitionsWithIndex(filfilter).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))\
    .mapValues(lambda x: x[0]/x[1]).map(lambda x: (x[0][0],(x[0][1],x[1]))).groupByKey().mapValues(outforma).sortBy(lambda x: x[0])

    header = sc.parallelize(['cbg_fips','2019-03','2019-10','2020-03','2020-10'])
    output1 = header.union(weeklypattern).saveAsTextFile('myOutput')