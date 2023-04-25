---
id: 7
title: "Pittsburgh Household Data"
subtitle: "Pandas for data manipulation"
date: "2018.5.10"
tags: "pandas, csv"
---



https://catalog.data.gov/dataset/pittsburgh-american-community-survey-2014-miscellaneous-data


```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    

# Lets use Pandas to extract and look at these CSVs for Pittsburgh Communities


```
import pandas as pd

# open average household income for the past twelve months
counties = pd.read_csv("/content/drive/My Drive/Data/Pittsburgh/aggregate-household-income-in-the-past-12-months-in-2014-inflation-adjusted-dollars.csv")

counties
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
      <th>Neighborhood</th>
      <th>Id</th>
      <th>Estimate; Aggregate household income in the past 12 months (in 2014 Inflation-adjusted dollars)</th>
      <th>Margin of Error; Aggregate household income in the past 12 months (in 2014 Inflation-adjusted dollars)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allegheny Center</td>
      <td>1</td>
      <td>28265700</td>
      <td>8176700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Allegheny West</td>
      <td>2</td>
      <td>23755400</td>
      <td>15159354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Allentown</td>
      <td>3</td>
      <td>40566300</td>
      <td>8358735.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arlington</td>
      <td>4</td>
      <td>33111400</td>
      <td>6248592.971</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arlington Heights</td>
      <td>5</td>
      <td>2515000</td>
      <td>1056475</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Upper Lawrenceville</td>
      <td>87</td>
      <td>60604600</td>
      <td>14808774.52</td>
    </tr>
    <tr>
      <th>87</th>
      <td>West End</td>
      <td>88</td>
      <td>4884400</td>
      <td>2447655</td>
    </tr>
    <tr>
      <th>88</th>
      <td>West Oakland</td>
      <td>89</td>
      <td>15820400</td>
      <td>3620802.954</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Westwood</td>
      <td>90</td>
      <td>90594300</td>
      <td>11234701.15</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Windgap</td>
      <td>91</td>
      <td>33450000</td>
      <td>6183325.504</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 4 columns</p>
</div>



So First two columns associate name of the neighborhood and ID number of each neighborhood.

Then each column after lists the Total Population, then those with salay income and without. 


