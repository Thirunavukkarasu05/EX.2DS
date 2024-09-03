# EXNO2DS
~~~
Name: THIRUNAVUKKARASU P 
Reg No: 212222040173
~~~
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![364001529-b7b0e093-ad83-4582-ad25-533d0a892bb2](https://github.com/user-attachments/assets/2bb93577-ca4e-4807-9903-2c6bdd3fc127)
```
dt.info()
```
![364001809-c209c169-b67a-43e7-b890-beead962faa2](https://github.com/user-attachments/assets/1913a03d-7dc9-4df2-87a1-7800fcbf246c)

```
dt.shape
```
![364001958-14969779-cb37-47c5-8685-713b62fb1d3e](https://github.com/user-attachments/assets/fa8b0224-7c88-468c-8513-7670dae36859)
```
#dt.set_index("PassengerId",inplace=True)
dt
```
![364002198-a9c396a2-3a76-465f-8b90-66e47c1c9aa7](https://github.com/user-attachments/assets/45793cf7-81c5-4959-b903-b721d32d9e2d)
```
dt.describe()
```
![364002413-a475f15e-b45e-4ca5-84f2-f9caad66ad88](https://github.com/user-attachments/assets/fdbc6b33-9ca5-4f0f-9520-dc4479c3e2fa)
```
dt.nunique()
```
![364002629-919af540-cdb6-4a80-967c-fc4c712fc8da](https://github.com/user-attachments/assets/93372520-3011-4439-9ac9-0f8a445f4cdc)
```
dt["Survived"].value_counts()
```
![364002816-9eae7390-2604-4309-9dd4-1a226c0abf83](https://github.com/user-attachments/assets/8ca0ce06-c34d-4d20-be50-17caf81f45f6)
```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![364002993-9343c988-d002-4498-91c9-d48dd780f4f0](https://github.com/user-attachments/assets/28c491f7-fad9-4555-997b-8e977e111de1)
```
sns.countplot(data=dt,x="Survived")
```
![364003222-b66c80d1-a8a3-41ef-a68b-c13583fbdfb4](https://github.com/user-attachments/assets/de78fb03-4021-4594-a179-e680607f04cf)
```
dt.Pclass.unique()
```
![364003446-695a6038-6167-4a73-8828-6a589af7a0a5](https://github.com/user-attachments/assets/1e2363c0-7039-4ecb-9c05-13faed5cb51b)
```
dt.rename(columns={'Sex': 'Gender'}, inplace = True)
dt
```
![364003678-2c18fc03-6645-4704-9822-5e5cfbf7f371](https://github.com/user-attachments/assets/3c7fc5a5-6480-4f42-baed-742dde88bb5c)
```
sns.catplot(x='Gender',col='Survived',kind='count',data=dt,height=5, aspect= 0.7)
```
![364003917-105a919c-276b-4dd8-b393-b19ba86bbc47](https://github.com/user-attachments/assets/11ba645b-4545-4c28-af81-7fac4a9bc704)
```
sns.catplot(x='Survived', hue='Gender',data=dt,kind='count')
```
![364004263-ba8a4378-461c-4974-a2d2-469e6866be91](https://github.com/user-attachments/assets/539a0823-197c-44db-8729-4c663b61b816)
```
dt.boxplot(column='Age',by='Survived')
```
![364004577-1df26b72-61c8-4143-a9cb-37b178591891](https://github.com/user-attachments/assets/348ac8d9-ee3d-4550-bc19-6c51a7993db9)
```
sns.scatterplot(x=dt['Age'], y= dt['Fare'])
```
![364004746-ee0a50d0-25cf-4a3d-8ceb-a6a0a071c9ae](https://github.com/user-attachments/assets/0e7369e5-23b0-4e48-be1a-a91361faf962)
```
sns.jointplot(x='Age',y='Fare',data=dt)
```
![364005011-3a000dbf-c4ec-48f7-980c-66956ec003db](https://github.com/user-attachments/assets/32cf61f9-dc08-4204-ae12-7be4d630d6c0)
```
sns.catplot(data=dt,col = 'Survived',x='Gender',hue='Pclass',kind='count')
```
![364005193-4e3c1c55-bcb4-42df-a6a1-47265bb27d61](https://github.com/user-attachments/assets/c0731b0b-2cc8-4aa0-bab2-ba13677b2f3c)
```
df = pd.read_csv('/content/titanic_dataset.csv')
df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()
sns.heatmap(corr,annot=True)
```
![364005336-8d16b9e5-9c9d-40a1-8beb-b6dc1ef1c788](https://github.com/user-attachments/assets/3d84bc2a-f3a0-401c-ad0d-a14ae7360a34)
```
sns.pairplot(dt)
```
![364005825-e21e20c4-9cf2-44e0-ab13-d6b30f544d13](https://github.com/user-attachments/assets/fe4e9f49-47cf-47b6-bd1f-d1e5a59b412f)
# RESULT
 Thus, the Exploratory Data Analysis on the given data set was performed successfully.
