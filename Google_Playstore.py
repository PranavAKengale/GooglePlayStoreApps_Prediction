import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import re
import plotly.express as px

df=pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Probability Project\googleplaystore.csv')

df.head(5)

To Find Null Values

df.shape

df.describe()





df.isnull()

#Plot of null values
plt.figure(figsize = (8,6))
sns.heatmap(df.isnull(), cbar=False , cmap = 'magma')

#Numerical representation
df.isnull().sum()

# Here we found that there is too much null value in Rating

df.drop(['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
##We dropped the unnecessary columns like..

df.dropna(inplace=True)
##We dropped unnecessay rows which were having null values

df.head(5)
##Our data set after dropping null values and unnessary rows and columns

df.shape
### Our data set now have 9366 rows and 9 columns

df['Size'] = df.Size.apply(lambda x: x.strip('+'))
df['Size'] = df.Size.apply(lambda x: x.replace(',', ''))
df['Size'] = df.Size.apply(lambda x: x.replace('M', ''))
df['Size'] = df.Size.apply(lambda x: x.replace('k', 'e-3'))
df['Size'] = df.Size.replace('Varies with device', 0)
df['Size'] = pd.to_numeric(df['Size'])

df.loc[df.Size == 0, 'Size'] = df.Size.median()
df.rename(columns={"Size": "Size_MB"}, inplace=True)


df['Installs'] = df.Installs.apply(lambda x: x.strip('+'))
df['Installs'] = df.Installs.apply(lambda x: x.replace(',', ''))
df['Installs'] = pd.to_numeric(df['Installs'])

df['Price']=df['Price'].apply(lambda x: x.replace('$',''))

###Our data after cleaning some unnecessay strings
df.head(5)

Data Visuallization to get a better understanding about our data


sns.set(rc={'figure.figsize':(7,6.8)})
plt.tight_layout()
plt.show()

plt.figure(figsize=[11, 4])
sns.set_context('talk')
sns.countplot(x='Rating', data = df,palette="YlOrBr")
plt.xticks(rotation=90)
plt.ylabel('Installs')
plt.show()

plt.figure(figsize=[8,6])
sns.countplot(x='Category',hue='Type',data=df,order=df.Category.value_counts().iloc[:21].index,palette='rocket') 
#plt.figure(figsize=[11, 20])
sns.color_palette("magma", as_cmap=True)
sns.set_context('talk')
plt.xticks(rotation=90,fontsize=9)
plt.ylabel('Installs')


Content_Ratings = df['Content Rating'].value_counts()
Content_Ratings
Figure = px.pie(labels=Content_Ratings.index, values=Content_Ratings.values, 
                title="Content Ratings", names=Content_Ratings.index,
                color_discrete_sequence=px.colors.sequential.RdBu)
Figure.update_traces(textposition='outside', textinfo='percent+label')
Figure.show()

plt.hist(x=df['Rating'],bins=30,density=True)

#Top 15 Genres and their No of installs
top_15 = df.groupby(['Genres']).agg({'Installs': "sum"})
df1 = top_15.reset_index()
df2 = df1.sort_values(by=['Installs'], ascending=False)
df3 = df2.reset_index()
df3.drop('index',axis = 1,inplace = True)
Genres = df3['Genres'].head(15)
Installs = df3['Installs'].head(15)
fig = plt.figure(figsize =(10,5))
sns.barplot(Genres, Installs)
plt.xticks(rotation=75)
plt.xlabel("Genres")
plt.ylabel("No. of Installs")
plt.title("Top 15 most installed Genres")
plt.show()


plt.figure(figsize=[11, 4])
sns.set_context('talk')
sns.countplot(x='Rating', data = df)
plt.xticks(rotation=65)
plt.show()

a=df.nlargest(100, 'Size_MB')['Category']
plt.figure(figsize=[7,6])
sns.countplot(x=a,data=df,palette='rocket') 
plt.xticks(rotation=90,fontsize=9)
plt.ylabel('Size_MB')

plt.hist(x=df['Size_MB'],bins=30,density=True)

from fitter import Fitter, get_common_distributions, get_distributions

f= Fitter(df["Size_MB"],
           distributions=["lognorm","norm"])
f.fit()
f.summary()

df.describe()

Inferential Statistical Analysis

# Hypothesis testing
import scipy.stats as stats

Two sampled T-test 

Paid_mean=df[df['Type']=='Paid']['Rating']

Free_mean=df[df['Type']=='Free']['Rating']

stats.ttest_ind(Paid_mean,Free_mean,equal_var=False)  

ttest,pval =stats.ttest_ind(Paid_mean,Free_mean,equal_var=False)
print("p-value",pval)
if pval <0.05:
  print("We reject null hypothesis")
else:
  print("We accept null hypothesis")

One Sampled T-Test

Family=df[df['Category']=='FAMILY']['Size_MB']

Category_mean=df['Size_MB'].mean()

tset, pval = stats.ttest_1samp(a=Family, popmean=Category_mean)
print('p-value',pval)
if pval < 0.05:    # alpha value is 0.05 or 5%
   print("We reject null hypothesis")
else:
  print("We accept null hypothesis")

Advanced Analytics

X=df[['Rating','Reviews','Size_MB','Price']]

y=df['Installs']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lr = LinearRegression(n_jobs=3)
lr.fit(X_train,y_train)
#ridge
r = Ridge(alpha=0.3)
r.fit(X_train,y_train)
#lars
lrs = Lars()
lrs.fit(X_train, y_train)
#lasso
ls = Lasso(alpha=0.3)
ls.fit(X_train, y_train)
#lasso lars
lslrs = LassoLars(alpha=0.3)
lslrs.fit(X_train, y_train)
# Random Forest Regressor
rfg=RandomForestRegressor()
rfg.fit(X_train,y_train)

models = {"Linear Regression":lr,"Ridge":r,"Lars":lrs,"Lasso":ls,"LassoLars":lslrs,"Random Forest Regressor":rfg}
train_mse = {}
val_mse = {}
for i,model in enumerate(models):
    print("Model: {}\n".format(model))
    
    pred = models[model].predict(X_train)
    print("Training Scores:")
    train_mse[model] = mean_squared_error(y_train,pred)
    print("Mean Sqaured Error: {}".format(train_mse[model]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_train,pred)))
    print("Explained Variance Score: {}".format(explained_variance_score(y_train,pred)))
    print("R2 Score: {}\n".format(r2_score(y_train,pred)))
    
    
    pred = models[model].predict(X_test)
    print("Validation Scores:")
    val_mse[model] = mean_squared_error(y_test,pred)
    print("Mean Sqaured Error: {}".format(val_mse[model]))
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,pred)))
    print("Explained Variance Score: {}".format(explained_variance_score(y_test,pred)))
    print("R2 Score: {}\n\n".format(r2_score(y_test,pred)))
    

model_train_error = pd.Series(data=list(train_mse.values()),index=list(train_mse.keys()))
model_val_error = pd.Series(data=list(val_mse.values()),index=list(val_mse.keys()))

fig= plt.figure(figsize=(10,6))
model_train_error.sort_values().plot.barh()
plt.show()


fig= plt.figure(figsize=(10,6))
model_val_error.sort_values().plot.barh()
plt.show()
