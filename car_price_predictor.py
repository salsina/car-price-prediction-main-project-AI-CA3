import pandas as pd
import numpy as  np
from __future__ import unicode_literals
from hazm import *
data = pd.read_csv("vehicles.csv")
del data['created_at']
data['description'] += " " + data['title']
del data['title']
data['year'].replace("<1366", 1365, inplace=True)

from __future__ import unicode_literals
from hazm import *
stopword_dict = {sw:0 for sw in stopwords_list()}
stemmer = Stemmer()
lemmatizer = Lemmatizer()
document = []
i=0
t = data[data['price']!= -1]
for desc in t['description']:
    if i == 30000:
        break
    sentence = ""
    tokenized_sentence = word_tokenize(desc)
    for word in tokenized_sentence:
        if word in  [':','؟','.',',','«','»','(',')','،','[',']','{','}','-','','؛', '\r', '\n']:
            continue
        if word in stopword_dict:
            continue
        w = stemmer.stem(word)
        lemmitized_word = lemmatizer.lemmatize(w)
        sentence += lemmitized_word + " "
    document.append(sentence)
    i+=1
    
from sklearn.feature_extraction.text import CountVectorizer
import collections
import itertools

vectorizer = CountVectorizer()
  
cv_fit = vectorizer.fit_transform(document)

inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}

count_vocab = cv_fit.toarray().sum(axis=0)

indexes = {}
for i in range(len(count_vocab)):
    indexes[count_vocab[i]] = i

tmp = collections.OrderedDict(sorted(indexes.items(), reverse = True))

tmp = dict(itertools.islice(tmp.items(), 10))

most_recurrent_words = []
for count in tmp:
    if tmp[count] in inv_map:
        most_recurrent_words.append(inv_map[tmp[count]])

most_recurrent_words.append('تمیز')
most_recurrent_words.pop(most_recurrent_words.index('لاستیک'))

for word in most_recurrent_words:
    repeats = []
    for row in data['description']:
        repeats.append(row.count(word))
    data[word] = repeats
del data['description']

brand_df = data.copy()[['brand','price']]
category_df = data.copy()[['category','price']]
brand_df.dropna(subset=['brand'],inplace=True)
mileage_df = data.copy()[['mileage','price']]
mileage_df.dropna(subset=['mileage'],inplace=True)
year_df = data.copy()[['year','price']]
year_df.dropna(subset=['year'],inplace=True)

category_df["category"] = category_df["category"].astype('category')
category_df["category"] = category_df["category"].cat.codes
brand_df["brand"] = brand_df["brand"].astype('category')
brand_df["brand"] = brand_df["brand"].cat.codes

data["category"] = data["category"].astype('category')
data["category"] = data["category"].cat.codes
data["brand"] = data["brand"].astype('category')
data["brand"] = data["brand"].cat.codes

import random 

def create_keys_ranges(df):
    counter = {}
    probs = {}
    keys_ranges = {}
    for row in df.iloc[:, 0]:
        if row in counter:
            counter[row] += 1
        else:
            counter[row] = 1
    
    for count in counter:
        probs[count] = counter[count]/len(df)
    
    prev_prob = 0
    for key in probs:
        keys_ranges[key] = [prev_prob, prev_prob + probs[key]]
        prev_prob += probs[key]
    
    return keys_ranges

brand_keys_ranges = create_keys_ranges(brand_df)
year_keys_ranges = create_keys_ranges(year_df)
mileage_keys_ranges = create_keys_ranges(mileage_df)

def fill_nan_randomly(keys_ranges):
    rand_num = random.random()
    for key in keys_ranges:
        if keys_ranges[key][0] < rand_num  and rand_num < keys_ranges[key][1]:
            return key

def fill_nan(col_name, col_keys_ranges):
    for row in range(len(data[col_name])):
        if data.at[row, col_name] == -1 or pd.isna(data.at[row, col_name]):
            num_to_replace = fill_nan_randomly(col_keys_ranges)
            data.at[row, col_name] = num_to_replace
      
    
fill_nan('brand', brand_keys_ranges)
fill_nan('mileage', mileage_keys_ranges)
fill_nan('year', year_keys_ranges)


y = pd.get_dummies(data.brand, prefix="brand")

for col in y.columns:
    data[col] = y[col]

del data["brand"]

test_data = data[data['price'] == -1]
train_data = data[data['price'] != -1]

import math

price_dict = {}
for price in train_data['price']:
    if price in price_dict:
        price_dict[price] += 1
    else:
        price_dict[price] = 1

root_types_num = len(price_dict)


def find_I(df):
    price_dict = {}
    for price in df['price']:
        if price in price_dict:
            price_dict[price] += 1
        else:
            price_dict[price] = 1
    
    length = len(df)
    ans = 0
    for i in price_dict:
        p = price_dict[i]/length
        ans += -p * math.log(p,root_types_num)
    return ans


root_entropy = find_I(train_data)

def find_info_gain(df, feature):
    df_dict = {}
    for row in df[feature]:
        if row in df_dict:
            df_dict[row] += 1
        else:
            df_dict[row] = 1
    df_length = len(df)
    final_ans = 0
    for i in df_dict:
        type_probability = df_dict[i]/df_length
        temp_df = df[df[feature] == i]
        ans = find_I(temp_df)
        final_ans += type_probability * ans
        
    return root_entropy - final_ans

info_gains = []
info_gains_cols = []
for col in train_data.columns:
    if col != 'price' and col[:5] != "brand":
        ig = find_info_gain(train_data, col)
        info_gains.append(ig)
        info_gains_cols.append(col)
        print(col,"Information Gain: ", ig)
        
import matplotlib.pyplot as plt
plt.figure(figsize =(15, 10))
plt.bar(info_gains_cols, info_gains)
plt.title('Information gain vs feature')
max_ig = max(info_gains)
idx = info_gains.index(max_ig)
print("max information gain is", max_ig, "which is for feature ",info_gains_cols[idx])

from sklearn import metrics

def rmse(ans, predicted):
    sum = 0
    n = len(predicted)
    count= 0
    for y in ans:
        a = y - predicted[count]
        sum += (a ** 2) / n
        count += 1
    return math.sqrt(sum)

def mse(ans, predicted):
    sum = 0
    n = len(predicted)
    count= 0
    for y in ans:
        a = y - predicted[count]
        sum += a ** 2
        count += 1
    return sum / n

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()
  
scaler.fit(train_data.drop('price', axis = 1))
scaled_features = scaler.transform(train_data.drop('price', axis = 1))
  
df_feat = pd.DataFrame(scaled_features, columns = train_data.columns[:-1])
df_feat.head()

X_train, X_test, y_train, y_test = train_test_split(scaled_features, train_data['price'], test_size = 0.30)

import seaborn as sns
import matplotlib.pyplot as plt

error_rate = []
  
for i in range(1, 32, 10):
    print(i)
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize =(10, 6))
plt.plot(range(1, 32,10), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
  
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors = 31)
  
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('RMSE: ', rmse(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))

from sklearn.linear_model import LinearRegression


y = train_data['price']
X = train_data.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('RMSE: ', rmse(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

from sklearn.linear_model import LinearRegression


y = train_data['price']
X = train_data.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('RMSE: ', rmse(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

from sklearn.linear_model import LinearRegression


y = train_data['price']
X = train_data.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('RMSE: ', rmse(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

X = train_data.drop('price', axis=1)
y = train_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

rmses = []
mses = []
for i in range(10,30):
    classifier = DecisionTreeClassifier(max_depth = i)
    classifier.fit(X_train, y_train)
    pred_i = classifier.predict(X_test)
    rmses.append(rmse(y_test, pred_i))
    mses.append(metrics.mean_squared_error(y_test, pred_i))
    
plt.plot(range(10,30), rmses)

plt.plot(range(10,30), mses)

from sklearn.tree import DecisionTreeClassifier

X = train_data.drop('price', axis=1)
y = train_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier(max_depth = 26)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('RMSE: ', rmse(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

y = train_data['price']
X = train_data.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('RMSE: ', rmse(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))