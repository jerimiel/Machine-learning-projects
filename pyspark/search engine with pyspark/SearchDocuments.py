#!/usr/bin/env python
# coding: utf-8

# In[2]:


import findspark
findspark.init()


# In[3]:


import pyspark


# In[4]:


from pyspark.sql import SparkSession


# In[5]:


spark=SparkSession.builder.getOrCreate()


# In[6]:


import os
lis=os.listdir(r"C:\Users\Sky\Videos\Data Science\new\train")


# In[7]:


df=[]

for i in range(0,10000):
    empty={'index':None,'content':None}
    a=open(r"C:\Users\Sky\Videos\Data Science\new\train"+'\\'+lis[i])
    empty['index']=lis[i].split('.')[0]
    empty['content']=a.read()
    df.append(empty)
    a.close()


# In[8]:


ans=spark.createDataFrame(df)


# In[9]:


ans.show()


# In[10]:


from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.sql.functions import udf,lit
from pyspark.sql.types import IntegerType,FloatType,DoubleType,ArrayType


# In[11]:


df=ans


# In[12]:


token=Tokenizer(inputCol='content',outputCol='tokenized')
stop=StopWordsRemover(inputCol='tokenized',outputCol='removed')
hashed=HashingTF(numFeatures=100000,inputCol='removed',outputCol='hashed')


# In[13]:


df=token.transform(df)
df=stop.transform(df)
df=hashed.transform(df)


# In[ ]:





# In[14]:


idf=IDF(inputCol='hashed',outputCol='idfed',minDocFreq=2).fit(df)


# In[15]:


df=idf.transform(df)


# In[16]:


df.printSchema()


# In[17]:


df.show(5)


# In[33]:


hashvalue=hashed.indexOf('electoral')


# In[34]:


hashvalue


# In[35]:


myudf=udf(lambda x:int(x[hashvalue]),IntegerType())


# In[36]:


new_df=df.withColumn('count',myudf(df['idfed']))


# In[37]:


new_df.printSchema()


# In[38]:


final=new_df.sort('count',ascending=False)


# In[41]:


final.show(4)


# In[44]:


item=final.select('content').collect()


# In[40]:


item[0]


# In[ ]:




