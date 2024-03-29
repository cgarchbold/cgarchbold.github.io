---
id: 3
title: "Mapping Kentucky County Populations"
subtitle: "Using geopandas to plot geo-spatial data"
date: "2018.10.04"
tags: "geopandas, matplotlib"
---

```
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```
pip install geopandas
```


```
import geopandas as gpd

counties = gpd.read_file("/content/drive/My Drive/Data/Kentucky/county_census/temp/county_census.shp")

counties.head()

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
      <th>NAME</th>
      <th>NAME2</th>
      <th>FIPS_NO</th>
      <th>FIPS_TXT</th>
      <th>CENPOP2000</th>
      <th>POPEST05</th>
      <th>POPEST10</th>
      <th>POPEST15</th>
      <th>POPEST20</th>
      <th>POPEST25</th>
      <th>POPEST30</th>
      <th>HHPOP90</th>
      <th>HH90</th>
      <th>POPPERHH90</th>
      <th>HHPOP95</th>
      <th>HH95</th>
      <th>POPPERHH95</th>
      <th>HHPOP00</th>
      <th>HH00</th>
      <th>POPPERHH00</th>
      <th>HHPOP05</th>
      <th>HH05</th>
      <th>POPPERHH05</th>
      <th>HHPOP10</th>
      <th>HH10</th>
      <th>POPPERHH10</th>
      <th>HHPOP15</th>
      <th>HH15</th>
      <th>POPPERHH15</th>
      <th>HHPOP20</th>
      <th>HH20</th>
      <th>POPPERHH20</th>
      <th>HHPOP25</th>
      <th>HH25</th>
      <th>POPPERHH25</th>
      <th>HHPOP30</th>
      <th>HH30</th>
      <th>POPPERHH30</th>
      <th>area</th>
      <th>len</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GREENUP</td>
      <td>Greenup</td>
      <td>89</td>
      <td>089</td>
      <td>36891</td>
      <td>37000</td>
      <td>37026</td>
      <td>37005</td>
      <td>36886</td>
      <td>36676</td>
      <td>36354</td>
      <td>36302</td>
      <td>13414</td>
      <td>2.71</td>
      <td>36373</td>
      <td>13975</td>
      <td>2.60</td>
      <td>36444</td>
      <td>14536</td>
      <td>2.51</td>
      <td>36553</td>
      <td>14880</td>
      <td>2.46</td>
      <td>36579</td>
      <td>15205</td>
      <td>2.41</td>
      <td>36558</td>
      <td>15358</td>
      <td>2.38</td>
      <td>36439</td>
      <td>15308</td>
      <td>2.38</td>
      <td>36229</td>
      <td>15220</td>
      <td>2.38</td>
      <td>35907</td>
      <td>15085</td>
      <td>2.38</td>
      <td>9.880911e+09</td>
      <td>519983.004395</td>
      <td>POLYGON ((5696927.001 4163203.493, 5697018.008...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MASON</td>
      <td>Mason</td>
      <td>161</td>
      <td>161</td>
      <td>16800</td>
      <td>16893</td>
      <td>17116</td>
      <td>17476</td>
      <td>17763</td>
      <td>18011</td>
      <td>18173</td>
      <td>16490</td>
      <td>6537</td>
      <td>2.52</td>
      <td>16479</td>
      <td>6692</td>
      <td>2.46</td>
      <td>16468</td>
      <td>6847</td>
      <td>2.41</td>
      <td>16561</td>
      <td>6804</td>
      <td>2.43</td>
      <td>16784</td>
      <td>6986</td>
      <td>2.40</td>
      <td>17144</td>
      <td>7182</td>
      <td>2.39</td>
      <td>17431</td>
      <td>7302</td>
      <td>2.39</td>
      <td>17679</td>
      <td>7406</td>
      <td>2.39</td>
      <td>17841</td>
      <td>7474</td>
      <td>2.39</td>
      <td>6.869176e+09</td>
      <td>353399.657348</td>
      <td>POLYGON ((5447610.998 4172645.259, 5448187.491...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CARROLL</td>
      <td>Carroll</td>
      <td>41</td>
      <td>041</td>
      <td>10155</td>
      <td>10334</td>
      <td>10631</td>
      <td>10993</td>
      <td>11304</td>
      <td>11559</td>
      <td>11749</td>
      <td>9147</td>
      <td>3505</td>
      <td>2.61</td>
      <td>9526</td>
      <td>3723</td>
      <td>2.56</td>
      <td>9905</td>
      <td>3940</td>
      <td>2.51</td>
      <td>10084</td>
      <td>4050</td>
      <td>2.49</td>
      <td>10381</td>
      <td>4210</td>
      <td>2.47</td>
      <td>10743</td>
      <td>4378</td>
      <td>2.45</td>
      <td>11054</td>
      <td>4505</td>
      <td>2.45</td>
      <td>11309</td>
      <td>4609</td>
      <td>2.45</td>
      <td>11499</td>
      <td>4686</td>
      <td>2.45</td>
      <td>3.828135e+09</td>
      <td>329873.908860</td>
      <td>POLYGON ((5040176.500 4156066.248, 5040339.502...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LEWIS</td>
      <td>Lewis</td>
      <td>135</td>
      <td>135</td>
      <td>14092</td>
      <td>13608</td>
      <td>13578</td>
      <td>13969</td>
      <td>14267</td>
      <td>14476</td>
      <td>14621</td>
      <td>12901</td>
      <td>4713</td>
      <td>2.74</td>
      <td>13391</td>
      <td>5068</td>
      <td>2.64</td>
      <td>13880</td>
      <td>5422</td>
      <td>2.56</td>
      <td>13396</td>
      <td>5327</td>
      <td>2.51</td>
      <td>13366</td>
      <td>5412</td>
      <td>2.47</td>
      <td>13757</td>
      <td>5621</td>
      <td>2.45</td>
      <td>14055</td>
      <td>5743</td>
      <td>2.45</td>
      <td>14264</td>
      <td>5828</td>
      <td>2.45</td>
      <td>14409</td>
      <td>5887</td>
      <td>2.45</td>
      <td>1.380964e+10</td>
      <td>750422.834077</td>
      <td>POLYGON ((5692349.493 4154522.005, 5692476.497...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRIMBLE</td>
      <td>Trimble</td>
      <td>223</td>
      <td>223</td>
      <td>8125</td>
      <td>9105</td>
      <td>10316</td>
      <td>11434</td>
      <td>12467</td>
      <td>13383</td>
      <td>14163</td>
      <td>6027</td>
      <td>2246</td>
      <td>2.68</td>
      <td>7049</td>
      <td>2692</td>
      <td>2.62</td>
      <td>8070</td>
      <td>3137</td>
      <td>2.57</td>
      <td>9050</td>
      <td>3555</td>
      <td>2.55</td>
      <td>10261</td>
      <td>4074</td>
      <td>2.52</td>
      <td>11379</td>
      <td>4542</td>
      <td>2.51</td>
      <td>12412</td>
      <td>4954</td>
      <td>2.51</td>
      <td>13328</td>
      <td>5320</td>
      <td>2.51</td>
      <td>14108</td>
      <td>5631</td>
      <td>2.51</td>
      <td>4.354949e+09</td>
      <td>296645.732761</td>
      <td>POLYGON ((5040176.500 4156066.248, 5040766.002...</td>
    </tr>
  </tbody>
</table>
</div>




```
data = counties.loc[:, ['geometry','POPEST25']].copy()

data.head()
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
      <th>geometry</th>
      <th>POPEST25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POLYGON ((5696927.001 4163203.493, 5697018.008...</td>
      <td>36676</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POLYGON ((5447610.998 4172645.259, 5448187.491...</td>
      <td>18011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POLYGON ((5040176.500 4156066.248, 5040339.502...</td>
      <td>11559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POLYGON ((5692349.493 4154522.005, 5692476.497...</td>
      <td>14476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POLYGON ((5040176.500 4156066.248, 5040766.002...</td>
      <td>13383</td>
    </tr>
  </tbody>
</table>
</div>




```
data.plot(column="POPEST25", legend=True,legend_kwds={'label': "Population by County", 'orientation': "horizontal"}, figsize = (20,20))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7030df6e80>




    
![png](/images/MappingKentuckyCountyPopulations_4_1.png)
    

