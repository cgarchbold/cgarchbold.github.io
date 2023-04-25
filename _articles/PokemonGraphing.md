---
id: 8
title: "Poken Graphing"
subtitle: "Opening Pokemon Data and Plotting"
date: "2018.6.01"
tags: "pandas, matplotlib"
---



# Lets look at the Pokemon Data

https://www.kaggle.com/rounakbanik/pokemon


```
from google.colab import drive
drive.mount('/content/gdrive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive
    


```
import pandas as pd
```


```
pokemon_data = pd.read_csv('/content/gdrive/My Drive/Data/Pokemon/pokemon.csv')

pokemon_data.head()
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
      <th>abilities</th>
      <th>against_bug</th>
      <th>against_dark</th>
      <th>against_dragon</th>
      <th>against_electric</th>
      <th>against_fairy</th>
      <th>against_fight</th>
      <th>against_fire</th>
      <th>against_flying</th>
      <th>against_ghost</th>
      <th>against_grass</th>
      <th>against_ground</th>
      <th>against_ice</th>
      <th>against_normal</th>
      <th>against_poison</th>
      <th>against_psychic</th>
      <th>against_rock</th>
      <th>against_steel</th>
      <th>against_water</th>
      <th>attack</th>
      <th>base_egg_steps</th>
      <th>base_happiness</th>
      <th>base_total</th>
      <th>capture_rate</th>
      <th>classfication</th>
      <th>defense</th>
      <th>experience_growth</th>
      <th>height_m</th>
      <th>hp</th>
      <th>japanese_name</th>
      <th>name</th>
      <th>percentage_male</th>
      <th>pokedex_number</th>
      <th>sp_attack</th>
      <th>sp_defense</th>
      <th>speed</th>
      <th>type1</th>
      <th>type2</th>
      <th>weight_kg</th>
      <th>generation</th>
      <th>is_legendary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['Overgrow', 'Chlorophyll']</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>49</td>
      <td>5120</td>
      <td>70</td>
      <td>318</td>
      <td>45</td>
      <td>Seed Pokémon</td>
      <td>49</td>
      <td>1059860</td>
      <td>0.7</td>
      <td>45</td>
      <td>Fushigidaneフシギダネ</td>
      <td>Bulbasaur</td>
      <td>88.1</td>
      <td>1</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>grass</td>
      <td>poison</td>
      <td>6.9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>['Overgrow', 'Chlorophyll']</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>62</td>
      <td>5120</td>
      <td>70</td>
      <td>405</td>
      <td>45</td>
      <td>Seed Pokémon</td>
      <td>63</td>
      <td>1059860</td>
      <td>1.0</td>
      <td>60</td>
      <td>Fushigisouフシギソウ</td>
      <td>Ivysaur</td>
      <td>88.1</td>
      <td>2</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>grass</td>
      <td>poison</td>
      <td>13.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['Overgrow', 'Chlorophyll']</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>100</td>
      <td>5120</td>
      <td>70</td>
      <td>625</td>
      <td>45</td>
      <td>Seed Pokémon</td>
      <td>123</td>
      <td>1059860</td>
      <td>2.0</td>
      <td>80</td>
      <td>Fushigibanaフシギバナ</td>
      <td>Venusaur</td>
      <td>88.1</td>
      <td>3</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>grass</td>
      <td>poison</td>
      <td>100.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['Blaze', 'Solar Power']</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>52</td>
      <td>5120</td>
      <td>70</td>
      <td>309</td>
      <td>45</td>
      <td>Lizard Pokémon</td>
      <td>43</td>
      <td>1059860</td>
      <td>0.6</td>
      <td>39</td>
      <td>Hitokageヒトカゲ</td>
      <td>Charmander</td>
      <td>88.1</td>
      <td>4</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>fire</td>
      <td>NaN</td>
      <td>8.5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>['Blaze', 'Solar Power']</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>5120</td>
      <td>70</td>
      <td>405</td>
      <td>45</td>
      <td>Flame Pokémon</td>
      <td>58</td>
      <td>1059860</td>
      <td>1.1</td>
      <td>58</td>
      <td>Lizardoリザード</td>
      <td>Charmeleon</td>
      <td>88.1</td>
      <td>5</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>fire</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Lets choose something to look at and analyze. Lets see the distribution of Pokemon types in Generation 1.


```
T_G_data = pokemon_data[['generation','type1','type2']]

T_G_data = T_G_data[T_G_data['generation'] == 1]

# Here you can notice that some pokemon have 2 types, meaning that type 2 will be populated with NaNs for pokemon with only one type.
T_G_data = T_G_data[['type1','type2']]

T_G_data
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
      <th>type1</th>
      <th>type2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>grass</td>
      <td>poison</td>
    </tr>
    <tr>
      <th>1</th>
      <td>grass</td>
      <td>poison</td>
    </tr>
    <tr>
      <th>2</th>
      <td>grass</td>
      <td>poison</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fire</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fire</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>146</th>
      <td>dragon</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>147</th>
      <td>dragon</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148</th>
      <td>dragon</td>
      <td>flying</td>
    </tr>
    <tr>
      <th>149</th>
      <td>psychic</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>150</th>
      <td>psychic</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>151 rows × 2 columns</p>
</div>



Yep! That's correct. There are 151 Pokemon in Generation 1.

We now need to take this data and make counts, we can do that using pandas.


```
type1 = T_G_data['type1'].value_counts()
type2 = T_G_data['type2'].value_counts()
```


```
type_counts = type1.add(type2,fill_value=0)
```


```
import matplotlib.pyplot as plt

plt.figure(figsize=(30,20))
#type_counts
plt.bar(type_counts.index.values, height = type_counts, width = 0.7)
```




    <BarContainer object of 18 artists>




    
![png](/images/PokemonGraphing_9_1.png)
    

