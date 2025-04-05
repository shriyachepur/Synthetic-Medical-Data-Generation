#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split


# In[27]:


df=pd.read_csv('data/Data_Entry_2017.csv')


# In[28]:


df=df[['Image Index', 'Finding Labels']]


# In[29]:


df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))


# In[30]:


df_filtered = df[df['Finding Labels'].apply(lambda x: any(label in ['No Finding','Infiltration'] for label in x))]


df_filtered['Label'] = df_filtered['Finding Labels'].apply(lambda x: 0 if 'No Finding' in x else 1)




print(df_filtered['Label'].value_counts())


# In[34]:


df_final = df_filtered.groupby('Label', group_keys=False).apply(
    lambda group: group if group.name == 1 else group.iloc[40000:]
)


# In[35]:


# In[16]:


image_names = set(df_final['Image Index'].tolist())


# In[62]:


source_dir = "data/images_012/images"


# In[63]:


destination_dir = "data/required_data"


# In[64]:


for file in os.listdir(source_dir):
    if file in image_names:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(destination_dir, file)
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")


# In[11]:


import os
file_count = sum(1 for entry in os.scandir(destination_dir) if entry.is_file())
print("Number of files:", file_count)


# In[11]:


df_final.to_csv('final data.csv', index=False)


# In[36]:


df_train, df_temp = train_test_split(
    df_final,
    test_size=0.30,
    stratify=df_final['Label'],
    random_state=42
)


# In[37]:


df_test, df_valid = train_test_split(
    df_temp,
    test_size=1/3,
    stratify=df_temp['Label'],
    random_state=42
)


# In[38]:


print("Train set shape:", df_train.shape)
print("Test set shape:", df_test.shape)
print("Validation set shape:", df_valid.shape)

print("\nTrain label distribution:\n", df_train['Label'].value_counts())
print("Test label distribution:\n", df_test['Label'].value_counts())
print("Validation label distribution:\n", df_valid['Label'].value_counts())


# In[40]:


source_dir = "data/required_data"
destination_dir = "data/train_data"


# In[39]:


df_train


# In[41]:


image_names = set(df_train['Image Index'].tolist())


# In[42]:


for file in os.listdir(source_dir):
    if file in image_names:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(destination_dir, file)
        shutil.copy2(src_path, dest_path)
        os.remove(src_path)
        print(f"Moved: {src_path} -> {dest_path}")


# In[43]:


import os
file_count = sum(1 for entry in os.scandir(destination_dir) if entry.is_file())
print("Number of files:", file_count)


# In[44]:


test_dir = "data/test_data"


# In[45]:


import os
file_count = sum(1 for entry in os.scandir(test_dir) if entry.is_file())
print("Number of files:", file_count)


# In[46]:


val_dir = "data/validation_data"


# In[47]:


import os
file_count = sum(1 for entry in os.scandir(val_dir) if entry.is_file())
print("Number of files:", file_count)


# In[ ]:




