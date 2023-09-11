#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis and Features Engineering:
# ## Expecting Car Pricing Project
# 
# # Importing the cleaned and nicly structured data

# In[2]:


# import libraries

import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


#importing the dataframe after the wrangling step

df = pd.read_csv("aut_cleaned1.csv")


# In[4]:


#Setting the option to show all the columns

pd.set_option("display.max_columns", None)

# Remove the unnamed col
df.drop("Unnamed: 0", axis =1 , inplace = True)

df.head()


# # Exploratory Data Analysis(EDA)
# 
# ## 1- Analyzing Individual Feature Patterns Using Visualization:
# 
# ### A) Continous Numerical Values
# 

# In[5]:


# knowing the types of col 
df.dtypes


# 
#  We have continous numerical value columns **(symboling, normalized-losses, wheel-base, length, width, height, curb-weight, engine-size, bore, stroke, compression-ratio, horsepower, peak-rpm, city-l@k, highway-l@k)**
# 

# In[6]:


# visulaizing one by one through scatter plot to see the corelation

# Starting with the engine-size
sns.regplot(x= "engine-size", y= "price", data = df)

# printing the correlatino factor
df[["engine-size","price"]].corr()


# In[7]:


# bore correlation
sns.regplot(x= "bore", y="price", data =df)

#corr
df[["bore","price"]].corr()


# In[8]:


#stroke
sns.regplot(x="stroke" , y= "price", data =df)
df[["stroke", "price"]].corr()


# In[9]:


#compression-ratio
sns.regplot(x="compression-ratio" , y= "price", data =df)
df[["compression-ratio", "price"]].corr()


# In[10]:


#horsepower
sns.regplot(x="horsepower" , y= "price", data =df)
df[["horsepower", "price"]].corr()


# In[11]:


#peak-rpm
sns.regplot(x="peak-rpm" , y= "price", data =df)
df[["peak-rpm", "price"]].corr()


# In[12]:


#city-l@k
sns.regplot(x="city-l@k" , y= "price", data =df)
df[["city-l@k", "price"]].corr()


# In[13]:


#highway-l@k
sns.regplot(x="highway-l@k", y= "price", data = df)
df[["highway-l@k", "price"]].corr()


# In[14]:


#wheel-base
sns.regplot(x="wheel-base", y= "price", data = df)
df[["wheel-base", "price"]].corr()


# ## Results:
# engine size, horsepower, city-l@k and highway-l@k have **positive correlation** with price.  
# bore, stroke, compression-ratio, peak-rpm and wheel-base have **week linear relation** with the price.  
# 

# ### B) Categorical Variables :
# there are  **make	num-of-doors	body-style	drive-wheels	engine-location	engine-type	num-of-cylinders	fuel-system** are categorical variables --   
# the best way to visualize is the boxplot

# In[15]:


df.select_dtypes("object")


# In[16]:


# make box plt viz
sns.boxplot(x="make",y= "price", data = df)
plt.pyplot.xticks(rotation=90) 
plt.pyplot.show()


# In[17]:


#num_of_doors
sns.boxplot(x="num-of-doors", y="price", data =df)


# In[18]:


# body-style box plt viz
sns.boxplot(x="body-style",y= "price", data = df)


# In[19]:


# drive-wheels box plt viz
sns.boxplot(x="drive-wheels",y= "price", data = df)


# In[20]:


# engine-location box plt viz
sns.boxplot(x="engine-location",y= "price", data = df)


# In[21]:


# num-of-cylinders box plt viz
sns.boxplot(x="num-of-cylinders",y= "price", data = df)


# In[22]:


# fuel-system box plt viz
sns.boxplot(x="fuel-system",y= "price", data = df)


# In[23]:


# fuel-type-diesel box plt viz (0 means gas, 1 means diesel)
sns.boxplot(x="fuel-type-diesel", y="price", data=df)


# In[24]:


# aspiration-std box plt viz (0 means turbo, 1 means std)
sns.boxplot(x="aspiration-std", y="price", data=df)


# ## Results:
# > **make** are obviously magnitude the price as "bmw, mercedes-benz, porsche" are the most expensive  
# **number of doors, body-style, aspiratio type, fuel type and fuel-system** have **no relation** to the price  
# **drive-wheels** shows that "rwd" is the most exp where both "fwd and 4wd" are approximetly same  
# **engine-location** rear are way more expensive than front but it's just 3 records so, is it reliable??   
# **num-of-cylinders** eight cylinders is more exp

# ## 2- Descitive Statistics: 
# **Ungerstanding the data better unsing EDA ( the target variable is the price col):  
# * Describe Methode ( Int or Obj)  
# >the count of that variable  
#     the mean  
#     the standard deviation (std)  
#     the minimum value  
#     he IQR (Interquartile Range: 25%, 50% and 75%)  
#     the maximum value  
# * value_counts.to_frame Methode for objects

# In[25]:


# Describe for the statistic values
df.describe()


# In[26]:


# describe the object values

df.describe(include= ["object"])


# In[27]:


# create a dataframe for value counts of num_of_doors

#counting the values for unique items
value_num_of_doors = df["num-of-doors"].value_counts().to_frame()

#renaming the col
value_num_of_doors.rename(columns={"num_of_doors": "value_counts"}, inplace=True)

#renaming the index
value_num_of_doors.index.name= "num_of_doors"

value_num_of_doors


# In[28]:


#creating for loop to do the previous for each object col
object_column= df.select_dtypes(include = ["object"]).columns

for column in object_column:
    value_count  = df[column].value_counts()
    print(f'Value counts for {column}:\n{value_count}\n')


# **we notice that Value counts for engine-location:**  
#  > front    198  
#    rear       3  
# **so, it's only 3 records for rear engine location, as a result we can't count on it as an independent variable for the price**

# # Correlation btween independant variables and the dependant price
# let's try the body_stle and drive_wheels and it's correlation with the price

# In[29]:


# create a df with specific col and the price
df_body_wheel = df[["drive-wheels","body-style","price"]]

# group by the drive wheel
df_body_wheel.groupby(["drive-wheels"]).mean()


# In[30]:


# group by the body style
df_body_wheel.groupby(["body-style"]).mean()


# ### Results:  
# * According to **drive-wheels** -- the **rwd** is the most expensive, where both **fwd & 4wd** are approximetly the same price  
# * According to **body-style** -- the **hardtop & convertible** are the most exp, where **htachback** is the cheapest.

# In[70]:


# let's group by both the previous
df_body_wheel_grouped = df_body_wheel.groupby(["drive-wheels", "body-style"], as_index= False).mean()

# let's make it more visible using pivoy methode
df_body_wheel_grouped_pivot = df_body_wheel_grouped.pivot(index = "drive-wheels", columns="body-style")

# fill the nan with 0 
df_pivot_body = df_body_wheel_grouped_pivot.fillna(0)


# In[32]:


# visualizing the correlation table with heatmap

sns.heatmap(df_pivot_body, cmap="RdBu")


# ### Results:
# > **rwd** is the most expensive 
# > **4wd--(convertrible & hardrtop)** is the cheapest
# > 

# ## 3- Correlations:
# #### A) Peasron Correlation
# Will use pearson correlation for the continous numerical values 
# ![Picture1.png.png](attachment:Picture1.png.png)
# 
# **Pearson coeffecient**   
#  <li><b>1</b>: Perfect positive linear correlation.</li>
#     <li><b>0</b>: No linear correlation, the two variables most likely do not affect each other.</li>
#     <li><b>-1</b>: Perfect negative linear correlation.</li>
# </ul>
# 
# **Probabilty(P-Value): how much the pearson corr reflects the population:**
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>
# 

# In[33]:


# peason coeffient numerical
df.corr()


# In[37]:


# pearson coeffient viz (heatmap)
sns.heatmap(df.corr(),cmap= "RdBu")
plt.pyplot.xticks(rotation= 90)
plt.pyplot.show()


# ## Results:  
# **from the heatmap:**  
# there is strong pos corr btw price and highway, city,horsepower, enginesize

# In[52]:


#impost stats
from scipy import stats

# calculate pearson coeff and p value for highway
per_cof_hi, p_value_hi = stats.pearsonr(df["highway-l@k"], df["price"])


# calculate pearson coeff and p value for city-l@k
per_cof_ct, p_value_ct = stats.pearsonr(df["city-l@k"], df["price"])

# calculate pearson coeff and p value for horsepower
per_cof_hp, p_value_hp = stats.pearsonr(df["horsepower"], df["price"])

# calculate pearson coeff and p value for engine-size
per_cof_ez, p_value_ez = stats.pearsonr(df["engine-size"], df["price"])

# calculate pearson coeff and p value for curb-weight
per_cof_cw, p_value_cw = stats.pearsonr(df["curb-weight"], df["price"])

# calculate pearson coeff and p value for bore
per_cof_bo, p_value_bo = stats.pearsonr(df["bore"], df["price"])

# printing table
print("|Variable\t|\t pear_coeff\t|\t P-Value\t|")
print("--------------------------------------------")
print(f'highway-l@k|\t {per_cof_hi}|\t {p_value_hi}|')
print(f'city-l@k|\t {per_cof_ct}|\t {p_value_ct}|')
print(f'horsepower|\t {per_cof_hp}|\t {p_value_hp}|')
print(f'engine-size|\t {per_cof_ez}|\t {p_value_ez}|')
print(f'curb-weight|\t {per_cof_cw}|\t {p_value_cw}|')
print(f'bore\t|\t {per_cof_bo}|\t {p_value_bo}|')


# ## Results: 
# We have **STRONG POS CORR** btw price and highway, city,horsepower, enginesize and crub weight and we are **CONFIEDENT** as p-value is way less than .001  
# where with **bore** we confident that there is no corr

# ## B) Chi^2 test - test of association
# will apply chi^2 test to correlate 2 categorical variables to understand is there a correlation btw or not.   
# Null Hypothesis(Ho) is the 2 var are independent 

# In[53]:


# let's check the categorical varibles
df.select_dtypes("object")


# In[76]:


# let's check if there is relation btw engine-type and fuel-system
chi_df = df[["engine-type", "fuel-system"]]

#found that there is 1 in the engine type let's drop it
df["engine-type"].unique()
df[df["engine-type"]== 'l'] # will leave it untill understand is it wrong entry or not

# back to the relation
chi_value_cou = chi_df.value_counts()

# Pivot the DataFrame
pivot_table = chi_value_cou.reset_index().pivot(index='fuel-system', columns='engine-type', values=0)

# Fill missing values with 0
pivot_table = pivot_table.fillna(0)

# Print the resulting table
print(pivot_table)

# apply the chi^2 test
chi2 = stats.chi2_contingency(pivot_table, correction= True)
chi2


# ## Results:
# **The Chi^2 test Value** is 207.1421113661918  
# **P-Value** is 2.5875550912835265e-26 *which way smaller than .05 so we reject the null hypotheseis, so the 2 variables are depedent*   
# **Degree of freedom** is 35

# In[ ]:




