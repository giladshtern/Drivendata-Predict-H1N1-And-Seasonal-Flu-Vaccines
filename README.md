# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

## Overview
Can you predict whether people got H1N1 and seasonal flu vaccines using information they shared about their backgrounds, opinions, and health behaviors?

In this challenge, we will take a look at vaccination, a key public health measure used to fight infectious diseases. Vaccines provide immunization for individuals, and enough immunization in a community can further reduce the spread of diseases through "herd immunity."

As of the launch of this competition, vaccines for the COVID-19 virus are still under development and not yet available. The competition will instead revisit the public health response to a different recent major respiratory disease pandemic. Beginning in spring 2009, a pandemic caused by the H1N1 influenza virus, colloquially named "swine flu," swept across the world. Researchers estimate that in the first year, it was responsible for between 151,000 to 575,000 deaths globally.

A vaccine for the H1N1 flu virus became publicly available in October 2009. In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts.

This is a practice competition designed to be accessible to participants at all levels. That makes it a great place to dive into the world of data science competitions. Come on in from the waiting room and try your (hopefully steady) hand at predicting vaccinations.


```python
#Step1 - Load Dataset
import nbconvert
import pandas as pd
train_df = pd.read_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/training_set_features.csv")
train_labels = pd.read_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/training_set_labels.csv")
test_df = pd.read_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/test_set_features.csv")
train_df.head()
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
      <th>respondent_id</th>
      <th>h1n1_concern</th>
      <th>h1n1_knowledge</th>
      <th>behavioral_antiviral_meds</th>
      <th>behavioral_avoidance</th>
      <th>behavioral_face_mask</th>
      <th>behavioral_wash_hands</th>
      <th>behavioral_large_gatherings</th>
      <th>behavioral_outside_home</th>
      <th>behavioral_touch_face</th>
      <th>...</th>
      <th>income_poverty</th>
      <th>marital_status</th>
      <th>rent_or_own</th>
      <th>employment_status</th>
      <th>hhs_geo_region</th>
      <th>census_msa</th>
      <th>household_adults</th>
      <th>household_children</th>
      <th>employment_industry</th>
      <th>employment_occupation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>Below Poverty</td>
      <td>Not Married</td>
      <td>Own</td>
      <td>Not in Labor Force</td>
      <td>oxchjgsf</td>
      <td>Non-MSA</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>Below Poverty</td>
      <td>Not Married</td>
      <td>Rent</td>
      <td>Employed</td>
      <td>bhuqouqj</td>
      <td>MSA, Not Principle  City</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>pxcmvdjn</td>
      <td>xgwztkwe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>&lt;= $75,000, Above Poverty</td>
      <td>Not Married</td>
      <td>Own</td>
      <td>Employed</td>
      <td>qufhixun</td>
      <td>MSA, Not Principle  City</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>rucpziij</td>
      <td>xtkaffoo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>Below Poverty</td>
      <td>Not Married</td>
      <td>Rent</td>
      <td>Not in Labor Force</td>
      <td>lrircsnp</td>
      <td>MSA, Principle City</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>&lt;= $75,000, Above Poverty</td>
      <td>Married</td>
      <td>Own</td>
      <td>Employed</td>
      <td>qufhixun</td>
      <td>MSA, Not Principle  City</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>wxleyezf</td>
      <td>emcorrxb</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
#Step2 - Check Col types, NA, unique.
df = pd.DataFrame(columns = ['Col', 'Type', 'NA', '%NA', 'UniqLen']) 
colList = list(train_df)
for i, value in enumerate(colList):
    df.loc[i] = [value, train_df.dtypes[i], train_df[value].isna().sum(),  train_df[value].isna().sum()/len(train_df), len(train_df[value].unique())]

df
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
      <th>Col</th>
      <th>Type</th>
      <th>NA</th>
      <th>%NA</th>
      <th>UniqLen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>respondent_id</td>
      <td>int64</td>
      <td>0</td>
      <td>0.000000</td>
      <td>26707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>h1n1_concern</td>
      <td>float64</td>
      <td>92</td>
      <td>0.003445</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>h1n1_knowledge</td>
      <td>float64</td>
      <td>116</td>
      <td>0.004343</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>behavioral_antiviral_meds</td>
      <td>float64</td>
      <td>71</td>
      <td>0.002658</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>behavioral_avoidance</td>
      <td>float64</td>
      <td>208</td>
      <td>0.007788</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>behavioral_face_mask</td>
      <td>float64</td>
      <td>19</td>
      <td>0.000711</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>behavioral_wash_hands</td>
      <td>float64</td>
      <td>42</td>
      <td>0.001573</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>behavioral_large_gatherings</td>
      <td>float64</td>
      <td>87</td>
      <td>0.003258</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>behavioral_outside_home</td>
      <td>float64</td>
      <td>82</td>
      <td>0.003070</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>behavioral_touch_face</td>
      <td>float64</td>
      <td>128</td>
      <td>0.004793</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>doctor_recc_h1n1</td>
      <td>float64</td>
      <td>2160</td>
      <td>0.080878</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>doctor_recc_seasonal</td>
      <td>float64</td>
      <td>2160</td>
      <td>0.080878</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>chronic_med_condition</td>
      <td>float64</td>
      <td>971</td>
      <td>0.036358</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>child_under_6_months</td>
      <td>float64</td>
      <td>820</td>
      <td>0.030704</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>health_worker</td>
      <td>float64</td>
      <td>804</td>
      <td>0.030104</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>health_insurance</td>
      <td>float64</td>
      <td>12274</td>
      <td>0.459580</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>opinion_h1n1_vacc_effective</td>
      <td>float64</td>
      <td>391</td>
      <td>0.014640</td>
      <td>6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>opinion_h1n1_risk</td>
      <td>float64</td>
      <td>388</td>
      <td>0.014528</td>
      <td>6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>opinion_h1n1_sick_from_vacc</td>
      <td>float64</td>
      <td>395</td>
      <td>0.014790</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>opinion_seas_vacc_effective</td>
      <td>float64</td>
      <td>462</td>
      <td>0.017299</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>opinion_seas_risk</td>
      <td>float64</td>
      <td>514</td>
      <td>0.019246</td>
      <td>6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>opinion_seas_sick_from_vacc</td>
      <td>float64</td>
      <td>537</td>
      <td>0.020107</td>
      <td>6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>age_group</td>
      <td>object</td>
      <td>0</td>
      <td>0.000000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>education</td>
      <td>object</td>
      <td>1407</td>
      <td>0.052683</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>race</td>
      <td>object</td>
      <td>0</td>
      <td>0.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>sex</td>
      <td>object</td>
      <td>0</td>
      <td>0.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>income_poverty</td>
      <td>object</td>
      <td>4423</td>
      <td>0.165612</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>marital_status</td>
      <td>object</td>
      <td>1408</td>
      <td>0.052720</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>rent_or_own</td>
      <td>object</td>
      <td>2042</td>
      <td>0.076459</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>employment_status</td>
      <td>object</td>
      <td>1463</td>
      <td>0.054780</td>
      <td>4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>hhs_geo_region</td>
      <td>object</td>
      <td>0</td>
      <td>0.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>31</th>
      <td>census_msa</td>
      <td>object</td>
      <td>0</td>
      <td>0.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>household_adults</td>
      <td>float64</td>
      <td>249</td>
      <td>0.009323</td>
      <td>5</td>
    </tr>
    <tr>
      <th>33</th>
      <td>household_children</td>
      <td>float64</td>
      <td>249</td>
      <td>0.009323</td>
      <td>5</td>
    </tr>
    <tr>
      <th>34</th>
      <td>employment_industry</td>
      <td>object</td>
      <td>13330</td>
      <td>0.499120</td>
      <td>22</td>
    </tr>
    <tr>
      <th>35</th>
      <td>employment_occupation</td>
      <td>object</td>
      <td>13470</td>
      <td>0.504362</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Step3 - Remove  50% NA cols 
train_df.drop(['employment_industry', 'employment_occupation'],inplace=True,axis=1)
test_df.drop(['employment_industry', 'employment_occupation'],inplace=True,axis=1)
```


```python
#Step4 - Dictionary
dic = {}
objList = []
colList = list(train_df)
#Create list of all object col
for i, value in enumerate(colList):
    #Per each object col, find unique values and add into list.
    if train_df[value].dtype == 'object':
        objList += list(train_df[value].unique())
        
#Remove duplicate from list and 'nan'
objList = list(set(objList))
objList.pop(0)

#Build dic with values
for i, value in enumerate(objList):
    dic[value] = (i + 3) * 4 - 1
    #Go over dic and replace strings into numeric
    train_df = train_df.replace(value, dic[value])
    test_df = test_df.replace(value, dic[value])

```


```python
#Step5 - Adding labels features into train_df
train_df["h1n1_vaccine"] = train_labels["h1n1_vaccine"]
train_df["seasonal_vaccine"] = train_labels["seasonal_vaccine"]
test_df["h1n1_vaccine"] = '0'
test_df["seasonal_vaccine"] = '0'
```


```python
#Step6 - Fill NAs
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
```


```python
#Step7 - Convert all objsct cols into numeric & numeric 64 into 32 format.
colList = list(train_df)
for i, value in enumerate(colList):
    if train_df[value].dtypes == 'int64':
        train_df[value] = train_df[value].astype('int32')
        test_df[value] = test_df[value].astype('int32')
    elif train_df[value].dtypes == 'float64':
        train_df[value] = train_df[value].astype('float32')
        test_df[value] = test_df[value].astype('float32')
    elif train_df[value].dtypes == 'object':
        train_df[value] = train_df[value].astype('int32')
        test_df[value] = test_df[value].astype('int32')
        
train_df.dtypes
train_df.to_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/training_features1.csv", index=False)
test_df.to_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/test_features1.csv", index=False)
```


```python
#Step8 - Check final columns type
train_df.dtypes
```




    respondent_id                    int32
    h1n1_concern                   float32
    h1n1_knowledge                 float32
    behavioral_antiviral_meds      float32
    behavioral_avoidance           float32
    behavioral_face_mask           float32
    behavioral_wash_hands          float32
    behavioral_large_gatherings    float32
    behavioral_outside_home        float32
    behavioral_touch_face          float32
    doctor_recc_h1n1               float32
    doctor_recc_seasonal           float32
    chronic_med_condition          float32
    child_under_6_months           float32
    health_worker                  float32
    health_insurance               float32
    opinion_h1n1_vacc_effective    float32
    opinion_h1n1_risk              float32
    opinion_h1n1_sick_from_vacc    float32
    opinion_seas_vacc_effective    float32
    opinion_seas_risk              float32
    opinion_seas_sick_from_vacc    float32
    age_group                        int32
    education                      float32
    race                             int32
    sex                              int32
    income_poverty                 float32
    marital_status                 float32
    rent_or_own                    float32
    employment_status              float32
    hhs_geo_region                   int32
    census_msa                       int32
    household_adults               float32
    household_children             float32
    h1n1_vaccine                     int32
    seasonal_vaccine                 int32
    dtype: object



## MAchine Learning - Pycaret
In this part, we will check the probability for both vaccination.
Let's run 2 experiments, the 1st for h1n1_vaccine & 2nd onr of seasonal_vaccine.


```python
#Step9 - Load Caret
from pycaret.classification import *

exp1 = setup(train_df, target = 'h1n1_vaccine')
```

     
    Setup Succesfully Completed!
    


<style  type="text/css" >
</style><table id="T_ea9d912a_de29_11ea_9a09_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow0_col1" class="data row0 col1" >1137</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow1_col0" class="data row1 col0" >Target Type</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow1_col1" class="data row1 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow2_col0" class="data row2 col0" >Label Encoded</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow2_col1" class="data row2 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow3_col0" class="data row3 col0" >Original Data</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow3_col1" class="data row3 col1" >(26707, 36)</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow4_col0" class="data row4 col0" >Missing Values </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow4_col1" class="data row4 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow5_col0" class="data row5 col0" >Numeric Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow5_col1" class="data row5 col1" >35</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow6_col0" class="data row6 col0" >Categorical Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow6_col1" class="data row6 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow7_col0" class="data row7 col0" >Ordinal Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow8_col0" class="data row8 col0" >High Cardinality Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow9_col0" class="data row9 col0" >High Cardinality Method </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow9_col1" class="data row9 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow10_col0" class="data row10 col0" >Sampled Data</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow10_col1" class="data row10 col1" >(2670, 36)</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow11_col1" class="data row11 col1" >(1868, 35)</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow12_col1" class="data row12 col1" >(802, 35)</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow13_col0" class="data row13 col0" >Numeric Imputer </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow13_col1" class="data row13 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow14_col0" class="data row14 col0" >Categorical Imputer </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow14_col1" class="data row14 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow15_col0" class="data row15 col0" >Normalize </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow15_col1" class="data row15 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow16_col0" class="data row16 col0" >Normalize Method </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow16_col1" class="data row16 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow17_col0" class="data row17 col0" >Transformation </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow18_col0" class="data row18 col0" >Transformation Method </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow18_col1" class="data row18 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow19_col0" class="data row19 col0" >PCA </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow20_col0" class="data row20 col0" >PCA Method </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow20_col1" class="data row20 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow21_col0" class="data row21 col0" >PCA Components </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow22_col0" class="data row22 col0" >Ignore Low Variance </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow22_col1" class="data row22 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow23_col0" class="data row23 col0" >Combine Rare Levels </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow23_col1" class="data row23 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow24_col0" class="data row24 col0" >Rare Level Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow24_col1" class="data row24 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow25_col0" class="data row25 col0" >Numeric Binning </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow25_col1" class="data row25 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow26_col0" class="data row26 col0" >Remove Outliers </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow26_col1" class="data row26 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow27_col0" class="data row27 col0" >Outliers Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow28_col0" class="data row28 col0" >Remove Multicollinearity </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow28_col1" class="data row28 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow29_col0" class="data row29 col0" >Multicollinearity Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow29_col1" class="data row29 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow30_col0" class="data row30 col0" >Clustering </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow30_col1" class="data row30 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow31_col0" class="data row31 col0" >Clustering Iteration </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow31_col1" class="data row31 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow32_col0" class="data row32 col0" >Polynomial Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow32_col1" class="data row32 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow33_col0" class="data row33 col0" >Polynomial Degree </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow34_col0" class="data row34 col0" >Trignometry Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow35_col0" class="data row35 col0" >Polynomial Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow36_col0" class="data row36 col0" >Group Features </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow37_col0" class="data row37 col0" >Feature Selection </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow38_col0" class="data row38 col0" >Features Selection Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow39_col0" class="data row39 col0" >Feature Interaction </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow40_col0" class="data row40 col0" >Feature Ratio </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ea9d912a_de29_11ea_9a09_94de8078c78elevel0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow41_col0" class="data row41 col0" >Interaction Threshold </td>
                        <td id="T_ea9d912a_de29_11ea_9a09_94de8078c78erow41_col1" class="data row41 col1" >None</td>
            </tr>
    </tbody></table>



```python
#Step10 - Compare modules
compare_models()
```




<style  type="text/css" >
    #T_089d804f_de2a_11ea_9388_94de8078c78e th {
          text-align: left;
    }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col1 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow0_col6 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col4 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow1_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow2_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col5 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow3_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow4_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col2 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow5_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow6_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow7_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow8_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow9_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col3 {
            background-color:  yellow;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow10_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow11_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow12_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow13_col6 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col0 {
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col1 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col2 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col3 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col4 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col5 {
            : ;
            text-align:  left;
        }    #T_089d804f_de2a_11ea_9388_94de8078c78erow14_col6 {
            : ;
            text-align:  left;
        }</style><table id="T_089d804f_de2a_11ea_9388_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col0" class="data row0 col0" >Extreme Gradient Boosting</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col1" class="data row0 col1" >0.871000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col2" class="data row0 col2" >0.886600</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col3" class="data row0 col3" >0.582300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col4" class="data row0 col4" >0.757300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col5" class="data row0 col5" >0.655500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow0_col6" class="data row0 col6" >0.578300</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col0" class="data row1 col0" >Extra Trees Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col1" class="data row1 col1" >0.867800</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col2" class="data row1 col2" >0.879200</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col3" class="data row1 col3" >0.526700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col4" class="data row1 col4" >0.786500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col5" class="data row1 col5" >0.626700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow1_col6" class="data row1 col6" >0.550700</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col0" class="data row2 col0" >CatBoost Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col1" class="data row2 col1" >0.867800</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col2" class="data row2 col2" >0.881000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col3" class="data row2 col3" >0.577400</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col4" class="data row2 col4" >0.742500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col5" class="data row2 col5" >0.647300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow2_col6" class="data row2 col6" >0.568000</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col0" class="data row3 col0" >Linear Discriminant Analysis</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col1" class="data row3 col1" >0.864000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col2" class="data row3 col2" >0.886500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col3" class="data row3 col3" >0.617600</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col4" class="data row3 col4" >0.705500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col5" class="data row3 col5" >0.656700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow3_col6" class="data row3 col6" >0.572800</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col0" class="data row4 col0" >Ridge Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col1" class="data row4 col1" >0.863500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col2" class="data row4 col2" >0.000000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col3" class="data row4 col3" >0.547100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col4" class="data row4 col4" >0.743700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col5" class="data row4 col5" >0.627600</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow4_col6" class="data row4 col6" >0.547000</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col0" class="data row5 col0" >Gradient Boosting Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col1" class="data row5 col1" >0.863500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col2" class="data row5 col2" >0.886900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col3" class="data row5 col3" >0.572300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col4" class="data row5 col4" >0.729500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col5" class="data row5 col5" >0.639200</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow5_col6" class="data row5 col6" >0.556900</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col0" class="data row6 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col1" class="data row6 col1" >0.860300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col2" class="data row6 col2" >0.868800</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col3" class="data row6 col3" >0.585100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col4" class="data row6 col4" >0.707400</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col5" class="data row6 col5" >0.638000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow6_col6" class="data row6 col6" >0.552900</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col0" class="data row7 col0" >Ada Boost Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col1" class="data row7 col1" >0.857100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col2" class="data row7 col2" >0.880100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col3" class="data row7 col3" >0.567100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col4" class="data row7 col4" >0.705000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col5" class="data row7 col5" >0.627000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow7_col6" class="data row7 col6" >0.540200</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col0" class="data row8 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col1" class="data row8 col1" >0.835700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col2" class="data row8 col2" >0.842200</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col3" class="data row8 col3" >0.604900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col4" class="data row8 col4" >0.613900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col5" class="data row8 col5" >0.608900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow8_col6" class="data row8 col6" >0.505000</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col0" class="data row9 col0" >Random Forest Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col1" class="data row9 col1" >0.831400</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col2" class="data row9 col2" >0.834500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col3" class="data row9 col3" >0.388300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col4" class="data row9 col4" >0.679900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col5" class="data row9 col5" >0.490700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow9_col6" class="data row9 col6" >0.399800</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col0" class="data row10 col0" >Naive Bayes</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col1" class="data row10 col1" >0.826000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col2" class="data row10 col2" >0.839200</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col3" class="data row10 col3" >0.665800</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col4" class="data row10 col4" >0.579200</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col5" class="data row10 col5" >0.618300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow10_col6" class="data row10 col6" >0.506500</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col0" class="data row11 col0" >Decision Tree Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col1" class="data row11 col1" >0.805700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col2" class="data row11 col2" >0.724300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col3" class="data row11 col3" >0.582400</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col4" class="data row11 col4" >0.542400</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col5" class="data row11 col5" >0.559000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow11_col6" class="data row11 col6" >0.435100</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col0" class="data row12 col0" >Logistic Regression</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col1" class="data row12 col1" >0.788000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col2" class="data row12 col2" >0.640700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col3" class="data row12 col3" >0.065500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col4" class="data row12 col4" >0.432100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col5" class="data row12 col5" >0.109100</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow12_col6" class="data row12 col6" >0.068200</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col0" class="data row13 col0" >K Neighbors Classifier</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col1" class="data row13 col1" >0.747900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col2" class="data row13 col2" >0.475600</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col3" class="data row13 col3" >0.083300</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col4" class="data row13 col4" >0.245500</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col5" class="data row13 col5" >0.122900</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow13_col6" class="data row13 col6" >0.014400</td>
            </tr>
            <tr>
                        <th id="T_089d804f_de2a_11ea_9388_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col0" class="data row14 col0" >SVM - Linear Kernel</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col1" class="data row14 col1" >0.514700</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col2" class="data row14 col2" >0.000000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col3" class="data row14 col3" >0.482600</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col4" class="data row14 col4" >0.230800</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col5" class="data row14 col5" >0.190000</td>
                        <td id="T_089d804f_de2a_11ea_9388_94de8078c78erow14_col6" class="data row14 col6" >-0.001300</td>
            </tr>
    </tbody></table>




```python
#Step:11 - Stacking model for improve ML
lda = create_model('lda')
gbc = create_model('gbc')
xgboost = create_model('xgboost')

# stacking models
stacker = stack_models(estimator_list = [xgboost ,lda,gbc], meta_model = xgboost)
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
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.9037</td>
      <td>0.9142</td>
      <td>0.6923</td>
      <td>0.8182</td>
      <td>0.7500</td>
      <td>0.6909</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8610</td>
      <td>0.9138</td>
      <td>0.6250</td>
      <td>0.6944</td>
      <td>0.6579</td>
      <td>0.5709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.8770</td>
      <td>0.8537</td>
      <td>0.5750</td>
      <td>0.7931</td>
      <td>0.6667</td>
      <td>0.5936</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.8717</td>
      <td>0.8910</td>
      <td>0.5750</td>
      <td>0.7667</td>
      <td>0.6571</td>
      <td>0.5802</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8556</td>
      <td>0.8753</td>
      <td>0.5750</td>
      <td>0.6970</td>
      <td>0.6301</td>
      <td>0.5415</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8503</td>
      <td>0.8446</td>
      <td>0.4250</td>
      <td>0.7727</td>
      <td>0.5484</td>
      <td>0.4676</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.8449</td>
      <td>0.8770</td>
      <td>0.5000</td>
      <td>0.6897</td>
      <td>0.5797</td>
      <td>0.4876</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.8182</td>
      <td>0.8160</td>
      <td>0.4000</td>
      <td>0.6154</td>
      <td>0.4848</td>
      <td>0.3804</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8441</td>
      <td>0.8751</td>
      <td>0.5641</td>
      <td>0.6471</td>
      <td>0.6027</td>
      <td>0.5063</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.8817</td>
      <td>0.9185</td>
      <td>0.6667</td>
      <td>0.7429</td>
      <td>0.7027</td>
      <td>0.6291</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.8608</td>
      <td>0.8779</td>
      <td>0.5598</td>
      <td>0.7237</td>
      <td>0.6280</td>
      <td>0.5448</td>
    </tr>
    <tr>
      <th>SD</th>
      <td>0.0227</td>
      <td>0.0316</td>
      <td>0.0901</td>
      <td>0.0622</td>
      <td>0.0731</td>
      <td>0.0840</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Step12 - Save experiment
save_experiment(experiment_name = 'G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/Exp1')
```

    Experiment Succesfully Saved
    

As we have finished with H1N1 model, let's continue with seasonal vaccine.


```python
#Step13 Create setup
exp2 = setup(train_df, target = 'seasonal_vaccine')
```

     
    Setup Succesfully Completed!
    


<style  type="text/css" >
</style><table id="T_165e0c38_de2a_11ea_ab10_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow0_col1" class="data row0 col1" >5919</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow1_col0" class="data row1 col0" >Target Type</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow1_col1" class="data row1 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow2_col0" class="data row2 col0" >Label Encoded</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow2_col1" class="data row2 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow3_col0" class="data row3 col0" >Original Data</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow3_col1" class="data row3 col1" >(26707, 36)</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow4_col0" class="data row4 col0" >Missing Values </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow4_col1" class="data row4 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow5_col0" class="data row5 col0" >Numeric Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow5_col1" class="data row5 col1" >35</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow6_col0" class="data row6 col0" >Categorical Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow6_col1" class="data row6 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow7_col0" class="data row7 col0" >Ordinal Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow8_col0" class="data row8 col0" >High Cardinality Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow9_col0" class="data row9 col0" >High Cardinality Method </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow9_col1" class="data row9 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow10_col0" class="data row10 col0" >Sampled Data</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow10_col1" class="data row10 col1" >(2670, 36)</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow11_col1" class="data row11 col1" >(1868, 35)</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow12_col1" class="data row12 col1" >(802, 35)</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow13_col0" class="data row13 col0" >Numeric Imputer </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow13_col1" class="data row13 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow14_col0" class="data row14 col0" >Categorical Imputer </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow14_col1" class="data row14 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow15_col0" class="data row15 col0" >Normalize </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow15_col1" class="data row15 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow16_col0" class="data row16 col0" >Normalize Method </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow16_col1" class="data row16 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow17_col0" class="data row17 col0" >Transformation </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow18_col0" class="data row18 col0" >Transformation Method </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow18_col1" class="data row18 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow19_col0" class="data row19 col0" >PCA </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow20_col0" class="data row20 col0" >PCA Method </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow20_col1" class="data row20 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow21_col0" class="data row21 col0" >PCA Components </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow22_col0" class="data row22 col0" >Ignore Low Variance </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow22_col1" class="data row22 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow23_col0" class="data row23 col0" >Combine Rare Levels </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow23_col1" class="data row23 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow24_col0" class="data row24 col0" >Rare Level Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow24_col1" class="data row24 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow25_col0" class="data row25 col0" >Numeric Binning </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow25_col1" class="data row25 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow26_col0" class="data row26 col0" >Remove Outliers </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow26_col1" class="data row26 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow27_col0" class="data row27 col0" >Outliers Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow28_col0" class="data row28 col0" >Remove Multicollinearity </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow28_col1" class="data row28 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow29_col0" class="data row29 col0" >Multicollinearity Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow29_col1" class="data row29 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow30_col0" class="data row30 col0" >Clustering </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow30_col1" class="data row30 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow31_col0" class="data row31 col0" >Clustering Iteration </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow31_col1" class="data row31 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow32_col0" class="data row32 col0" >Polynomial Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow32_col1" class="data row32 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow33_col0" class="data row33 col0" >Polynomial Degree </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow34_col0" class="data row34 col0" >Trignometry Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow35_col0" class="data row35 col0" >Polynomial Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow36_col0" class="data row36 col0" >Group Features </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow37_col0" class="data row37 col0" >Feature Selection </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow38_col0" class="data row38 col0" >Features Selection Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow39_col0" class="data row39 col0" >Feature Interaction </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow40_col0" class="data row40 col0" >Feature Ratio </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_165e0c38_de2a_11ea_ab10_94de8078c78elevel0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow41_col0" class="data row41 col0" >Interaction Threshold </td>
                        <td id="T_165e0c38_de2a_11ea_ab10_94de8078c78erow41_col1" class="data row41 col1" >None</td>
            </tr>
    </tbody></table>



```python
#Step14 - Compare modules
compare_models()
```




<style  type="text/css" >
    #T_36c97681_de2a_11ea_bc25_94de8078c78e th {
          text-align: left;
    }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col1 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col5 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col6 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col2 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col3 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col4 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col4 {
            background-color:  yellow;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col6 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col0 {
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col1 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col2 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col3 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col4 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col5 {
            : ;
            text-align:  left;
        }    #T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col6 {
            : ;
            text-align:  left;
        }</style><table id="T_36c97681_de2a_11ea_bc25_94de8078c78e" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col0" class="data row0 col0" >Extreme Gradient Boosting</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col1" class="data row0 col1" >0.806700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col2" class="data row0 col2" >0.880000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col3" class="data row0 col3" >0.770100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col4" class="data row0 col4" >0.807900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col5" class="data row0 col5" >0.787600</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow0_col6" class="data row0 col6" >0.610600</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col0" class="data row1 col0" >CatBoost Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col1" class="data row1 col1" >0.806200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col2" class="data row1 col2" >0.877700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col3" class="data row1 col3" >0.770100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col4" class="data row1 col4" >0.806800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col5" class="data row1 col5" >0.787300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow1_col6" class="data row1 col6" >0.609500</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col0" class="data row2 col0" >Gradient Boosting Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col1" class="data row2 col1" >0.805700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col2" class="data row2 col2" >0.880400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col3" class="data row2 col3" >0.770100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col4" class="data row2 col4" >0.806200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col5" class="data row2 col5" >0.786700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow2_col6" class="data row2 col6" >0.608500</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col0" class="data row3 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col1" class="data row3 col1" >0.805700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col2" class="data row3 col2" >0.872700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col3" class="data row3 col3" >0.771300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col4" class="data row3 col4" >0.804800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col5" class="data row3 col5" >0.786900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow3_col6" class="data row3 col6" >0.608500</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col0" class="data row4 col0" >Ridge Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col1" class="data row4 col1" >0.803500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col2" class="data row4 col2" >0.000000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col3" class="data row4 col3" >0.726400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col4" class="data row4 col4" >0.831200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col5" class="data row4 col5" >0.774800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow4_col6" class="data row4 col6" >0.602000</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col0" class="data row5 col0" >Linear Discriminant Analysis</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col1" class="data row5 col1" >0.803500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col2" class="data row5 col2" >0.869500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col3" class="data row5 col3" >0.726400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col4" class="data row5 col4" >0.831200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col5" class="data row5 col5" >0.774800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow5_col6" class="data row5 col6" >0.602000</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col0" class="data row6 col0" >Extra Trees Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col1" class="data row6 col1" >0.802500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col2" class="data row6 col2" >0.872800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col3" class="data row6 col3" >0.746000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col4" class="data row6 col4" >0.815900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col5" class="data row6 col5" >0.778400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow6_col6" class="data row6 col6" >0.600900</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col0" class="data row7 col0" >Ada Boost Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col1" class="data row7 col1" >0.800300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col2" class="data row7 col2" >0.873900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col3" class="data row7 col3" >0.756300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col4" class="data row7 col4" >0.804700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col5" class="data row7 col5" >0.778600</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow7_col6" class="data row7 col6" >0.597200</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col0" class="data row8 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col1" class="data row8 col1" >0.775100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col2" class="data row8 col2" >0.839400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col3" class="data row8 col3" >0.743700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col4" class="data row8 col4" >0.767000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col5" class="data row8 col5" >0.754900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow8_col6" class="data row8 col6" >0.547300</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col0" class="data row9 col0" >Random Forest Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col1" class="data row9 col1" >0.768200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col2" class="data row9 col2" >0.840100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col3" class="data row9 col3" >0.690800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col4" class="data row9 col4" >0.787600</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col5" class="data row9 col5" >0.735100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow9_col6" class="data row9 col6" >0.530700</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col0" class="data row10 col0" >Naive Bayes</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col1" class="data row10 col1" >0.763400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col2" class="data row10 col2" >0.822800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col3" class="data row10 col3" >0.708000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col4" class="data row10 col4" >0.767400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col5" class="data row10 col5" >0.735900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow10_col6" class="data row10 col6" >0.522300</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col0" class="data row11 col0" >Decision Tree Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col1" class="data row11 col1" >0.712000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col2" class="data row11 col2" >0.711500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col3" class="data row11 col3" >0.704600</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col4" class="data row11 col4" >0.687900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col5" class="data row11 col5" >0.695200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow11_col6" class="data row11 col6" >0.422500</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col0" class="data row12 col0" >Logistic Regression</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col1" class="data row12 col1" >0.631700</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col2" class="data row12 col2" >0.677000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col3" class="data row12 col3" >0.567800</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col4" class="data row12 col4" >0.612200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col5" class="data row12 col5" >0.588100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow12_col6" class="data row12 col6" >0.256200</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col0" class="data row13 col0" >K Neighbors Classifier</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col1" class="data row13 col1" >0.543400</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col2" class="data row13 col2" >0.545300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col3" class="data row13 col3" >0.510300</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col4" class="data row13 col4" >0.511500</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col5" class="data row13 col5" >0.510100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow13_col6" class="data row13 col6" >0.082600</td>
            </tr>
            <tr>
                        <th id="T_36c97681_de2a_11ea_bc25_94de8078c78elevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col0" class="data row14 col0" >SVM - Linear Kernel</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col1" class="data row14 col1" >0.524100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col2" class="data row14 col2" >0.000000</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col3" class="data row14 col3" >0.424100</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col4" class="data row14 col4" >0.634200</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col5" class="data row14 col5" >0.312900</td>
                        <td id="T_36c97681_de2a_11ea_bc25_94de8078c78erow14_col6" class="data row14 col6" >0.035800</td>
            </tr>
    </tbody></table>




```python
#Step:15 - Stacking model for improve ML
catboost = create_model('catboost')
lda = create_model('lda')
xgboost = create_model('xgboost')

# stacking models
stacker = stack_models(estimator_list = [xgboost ,lda,gbc], meta_model = xgboost)
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
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8182</td>
      <td>0.9018</td>
      <td>0.8276</td>
      <td>0.7912</td>
      <td>0.8090</td>
      <td>0.6357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7647</td>
      <td>0.8434</td>
      <td>0.7356</td>
      <td>0.7529</td>
      <td>0.7442</td>
      <td>0.5264</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.7701</td>
      <td>0.8362</td>
      <td>0.7701</td>
      <td>0.7444</td>
      <td>0.7571</td>
      <td>0.5389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.8021</td>
      <td>0.8808</td>
      <td>0.7931</td>
      <td>0.7841</td>
      <td>0.7886</td>
      <td>0.6027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8075</td>
      <td>0.8685</td>
      <td>0.7241</td>
      <td>0.8400</td>
      <td>0.7778</td>
      <td>0.6096</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8128</td>
      <td>0.8931</td>
      <td>0.7816</td>
      <td>0.8095</td>
      <td>0.7953</td>
      <td>0.6230</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.7594</td>
      <td>0.8457</td>
      <td>0.7126</td>
      <td>0.7561</td>
      <td>0.7337</td>
      <td>0.5146</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.8342</td>
      <td>0.8929</td>
      <td>0.7586</td>
      <td>0.8684</td>
      <td>0.8098</td>
      <td>0.6641</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8172</td>
      <td>0.8498</td>
      <td>0.7586</td>
      <td>0.8354</td>
      <td>0.7952</td>
      <td>0.6308</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.7903</td>
      <td>0.8839</td>
      <td>0.7356</td>
      <td>0.8000</td>
      <td>0.7665</td>
      <td>0.5768</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.7977</td>
      <td>0.8696</td>
      <td>0.7598</td>
      <td>0.7982</td>
      <td>0.7777</td>
      <td>0.5923</td>
    </tr>
    <tr>
      <th>SD</th>
      <td>0.0242</td>
      <td>0.0229</td>
      <td>0.0331</td>
      <td>0.0389</td>
      <td>0.0252</td>
      <td>0.0483</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Step16 - Save 2nd experiment
save_experiment(experiment_name = 'G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/Exp2')
```

    Experiment Succesfully Saved
    


```python
#Step17 - Load exp1 for predict H1N1 probability.
exp1 = load_experiment('G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/Exp1')
prediction1 = predict_model(stacker, data = test_df)
```


```python
#Step18 - Load exp1 for predict  seasonal probability.
exp2 = load_experiment('G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/Exp2')
prediction2 = predict_model(stacker, data = test_df)
```


```python
#Step19 - Build submission
submission = pd.DataFrame(columns=['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine'])
submission['respondent_id'] = test_df['respondent_id']
submission['h1n1_vaccine'] = prediction1['Score']
submission['seasonal_vaccine'] = prediction2['Score']
submission.head(30)
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
      <th>respondent_id</th>
      <th>h1n1_vaccine</th>
      <th>seasonal_vaccine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26707</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26708</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26709</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26710</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26711</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26712</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>6</th>
      <td>26713</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26714</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>8</th>
      <td>26715</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>9</th>
      <td>26716</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>10</th>
      <td>26717</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26718</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>12</th>
      <td>26719</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26720</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>14</th>
      <td>26721</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>15</th>
      <td>26722</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>16</th>
      <td>26723</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>17</th>
      <td>26724</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>18</th>
      <td>26725</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>19</th>
      <td>26726</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>20</th>
      <td>26727</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>21</th>
      <td>26728</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>22</th>
      <td>26729</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>23</th>
      <td>26730</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>24</th>
      <td>26731</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26732</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26733</td>
      <td>0.0622</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <th>27</th>
      <td>26734</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>28</th>
      <td>26735</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
    <tr>
      <th>29</th>
      <td>26736</td>
      <td>0.0764</td>
      <td>0.0764</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv("G:/DataScienceProject/Drivendata-Predict-H1N1-And-Seasonal-Flu-Vaccines/submit1.csv", index=False)
```
