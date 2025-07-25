---
title: Hennepin County Dataset
published: 2023-01-06
description: 'A geospatial dataset of parcels and their market value'
image: './hennepin_example.jpg'
tags: [dataset, geospatial, remote-sensing]
category: 'works'
draft: false 
---

[Link to Repository](https://github.com/cgarchbold/GISGather)

The Hennepin County Dataset was collected using GIS Open Data for Hennepin County, Minnesota. For each sample, a sub-region of fine-level aerial imagery from the year 2020 was cropped. 1915 images of size 302x302 were collected with a Ground Sample Distance (GSD) of 1 meter. 

Geometries were extracted from GIS Open Data for parcels. For each chip, corresponding parcels fully contained within each chip were extracted. We also collected the parcels' market value, which was accurate as of 2020. In order to effectively use the dataset for a regression task, we cleaned outliers and defects from the collected dataset. For instance, many of the original parcels in the Hennepin dataset included a market value of zero. These regions are often either unlabeled or public land. We also remove parcels with disproportionately high value from downtown Minneapolis. 

Each sample of the dataset contains the information: region image, parcel masks, and corresponding market values. We collected over 65,000 parcels with an average area of 984 pixels and an average value of 286346. The average value of a pixel on the training set is $379.74. We hold out 10% of the dataset for validation and 10% of the dataset for testing.
