---
id: 5
title: "Mapping Kentucky Forrests"
subtitle: "Using geopandas to plot geo-spatial data"
date: "2018.8.08"
tags: "geopandas, matplotlib"
---


Grabbing data from google drive


```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

## Import Data and extract Pandas DataFrame


```
import geopandas as gpd

path = "/content/drive/My Drive/Data/Woodlands2016/Woodlands2016.shp"

#reading data using geopandas
full_data = gpd.read_file(path)


#view the top (first five rows)
full_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OBJECTID</th>
      <th>SHAPE_Leng</th>
      <th>Shape_Le_1</th>
      <th>Shape_Area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>287443</td>
      <td>1054.643958</td>
      <td>1054.643958</td>
      <td>6.085877e+04</td>
      <td>POLYGON ((5206728.612 4305311.614, 5206708.345...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>287442</td>
      <td>516.257175</td>
      <td>516.257175</td>
      <td>1.797931e+04</td>
      <td>POLYGON ((5205657.338 4305400.221, 5205633.798...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>304386</td>
      <td>15077.412517</td>
      <td>15077.412517</td>
      <td>2.400849e+06</td>
      <td>POLYGON ((5204878.171 4305337.325, 5204892.193...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>287441</td>
      <td>940.564755</td>
      <td>940.564755</td>
      <td>4.244522e+04</td>
      <td>POLYGON ((5207534.783 4305202.197, 5207487.267...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>287440</td>
      <td>851.483539</td>
      <td>851.483539</td>
      <td>3.446043e+04</td>
      <td>POLYGON ((5208346.698 4305006.344, 5208285.331...</td>
    </tr>
  </tbody>
</table>
</div>



## Grab only the geometry data


```
data = full_data.loc[:, 'geometry'].copy()

data.head()
```




    0    POLYGON ((5206728.612 4305311.614, 5206708.345...
    1    POLYGON ((5205657.338 4305400.221, 5205633.798...
    2    POLYGON ((5204878.171 4305337.325, 5204892.193...
    3    POLYGON ((5207534.783 4305202.197, 5207487.267...
    4    POLYGON ((5208346.698 4305006.344, 5208285.331...
    Name: geometry, dtype: geometry



## Plotting


```
data.plot(figsize = (20,20))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f92aafc7630>




    
![png](/images/MappingKentuckyForrests_7_1.png)
    

