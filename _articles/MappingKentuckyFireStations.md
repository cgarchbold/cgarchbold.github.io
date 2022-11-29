---
id: 3
title: "Mapping Kentucky Fire Stations"
subtitle: "  "
date: "2019.1.20"
tags: "pandas, geopandas, matplotlib"
---

```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    


```
import geopandas as gpd

counties = gpd.read_file("/content/drive/My Drive/Data/Lexington/county/County.shp")

addresses = gpd.read_file("/content/drive/My Drive/Data/Lexington/addresspoint/AddressPoint.shp")

firestations = gpd.read_file("/content/drive/My Drive/Data/Lexington/firestation/FireStation.shp")
```


```

# Define a base map with county boundaries
ax = counties.plot(figsize=(10,10), color='none', edgecolor='silver', zorder=3)

addresses.plot(color='grey', markersize=1, ax=ax)
firestations.plot(color='red', markersize=24, ax=ax)

#assume some sort of average range for each firestation
firestations.plot(color='red', markersize=4000, ax=ax , edgecolor='red',facecolor='none',)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5db99082e8>




    
![png](/images/MappingKentuckyFireStations_2_1.png)
    

