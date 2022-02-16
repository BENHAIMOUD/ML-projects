#!/usr/bin/env python
# coding: utf-8

# # Sentiment analysis:Basic Needs Basic Rights Kenya 
# Created by <span style='color:blue; font-weight: bold;'>Mr Ayoub BENHAIMOUD</span>

# In[60]:


import pandas as pd


# In[31]:


mentalHealth = pd.read_csv('data/Train.csv')


# To explore this data use dome routine fonctions

# In[32]:


mentalHealth.head()


# As you can seed data includes (ID , text , label)

# In[33]:


mentalHealth.shape


# In[34]:


mentalHealth.label.value_counts()


# In[35]:


mentalHealth.info()


# <span style='color:blue; font-weight: bold;'> ===================Data preparation ==================== </span>

# ###### Features extraction

# <span style='font-weight:bold'>Bag-of-words :</span> transform each comment to a words counter vector

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


# Initialize a CountVectorizer object: count_vectorizer
bag_of_words = CountVectorizer(stop_words="english", analyzer='word') #max_features


# In[37]:


# Transforms the data into a bag of words
mentalHealth.dropna(subset=['text','ID'],inplace=True)
data_words = bag_of_words.fit_transform(products.text)


# In[38]:


print(bag_of_words.get_feature_names()[400:410])


# In[39]:


data_words.shape


# #### matrix of frequences of words

# In[40]:


pd.DataFrame(data_words.toarray()).head(3)


# #### Display the words

# In[41]:


bag_of_words.vocabulary_


# Get common stopwords that are often removed during preprocessing of text data

# In[43]:


print(bag_of_words.get_stop_words() )


# In[46]:


#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer()
#tfidf_data = vectorizer.fit_transform(data_words)


# <span style='color:blue; font-weight: bold;'> =================== Data preparation end ==================== </span>

# #### Build a sentiment classifier

# Train the sentiment classifier using SVM

# Split the data set to train and test set using fraction (80% and 20%), you may chnage it

# In[48]:


x=data_words
y=products['label']


# In[49]:


print('data  shape :', x.shape)
print('label  shape :', len(y))


# ##### Split the data set to train and test set using fraction (80% and 20%), you may chnage it
# 

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)


# ##### Instantiate SVC as classifier of SVM algorithm

# In[51]:


from sklearn.svm import SVC  


# In[52]:


svclassifier = SVC()  # probability=True, class_weight='balanced'


# In[53]:


svclassifier.fit(X_train, y_train)


# In[54]:


svclassifier.classes_


# In[55]:


y_pred = svclassifier.predict(X_test) # svclassifier.predict_proba


# ##### Evaluate the SVM model

# In[56]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[57]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ##### Applying the built model to compare between gotten prediction and reality

# In[58]:


y_pred[93]


# In[59]:


products['label'][93]


# <span style='color:blue; font-weight: bold;'> =================== End ==================== </span>
