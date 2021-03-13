#Data Types
df.dtypes
# check the data type of data frame "df" by .dtypes
print(df.dtypes)

#Describe
dataframe.describe()
df.describe()

# describe all the columns in "df" 
df.describe(include = "all")

#You can select the columns of a data frame by indicating the name of each column, for example, you can select the three columns as follows:
#dataframe[[' column 1 ',column 2', 'column 3']]
#Where "column" is the name of the column, you can apply the method ".describe()" to get the statistics of those columns as follows:
#dataframe[[' column 1 ',column 2', 'column 3'] ].describe()
#Apply the method to ".describe()" to the columns 'length' and 'compression-ratio'.

df[['length', 'compression-ratio']].describe()

#Info
dataframe.info()
# look at the info of "df"
df.info()








