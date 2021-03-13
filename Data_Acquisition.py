#import panad
import pandas as pd
import numpy as np

#Read data 
other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(other_path, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head(5)
# show last 10 rows 
print("The last 10 rows of the dataframe")
df.tail(10)

#add Headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
df.columns = headers
df.head(10)

#Replacing "?" with NaN values 
df1=df.replace('?',np.NaN)
df=df1.dropna(subset=["price"], axis=0)
df.head(20)

#retreving cloumns names
print(df.columns)

#Saving the Dataset
df.to_csv("automobile.csv", index=False)






























