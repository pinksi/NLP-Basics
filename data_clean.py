# NLP basics: reading and cleaning data
import pandas as pd
import string
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
# Reading data using open()
# rawData = open('SMSSpamCollection').read()

# # print(rawData[0:500])
# parseData = rawData.replace('\t', '\n').split('\n')
# labels = parseData[0::2]
# texts = parseData[1::2]
# # convert these data into dataframe
# data_df = pd.DataFrame({
#                 'labels': labels[:-1],
#                 'texts': texts,
#             })
# Reg=ular Expression
# [0-9] => search numbers from 0 to 9, single strings like 0, 1, 2..
# [0-9]+ => search numbers from 0 to 9 including multiple characters like 0, 123, 12..
# \s => check single whitespace
# \s+ => check multiple whitespaces
# \w+ => search for any non-word character and remove it
# can use re.split('\w+') or re.findall('\w+')
#===========================================================================================
data = pd.read_csv('SMSSpamCollection', sep='\t', header=None)
data.columns = ['labels', 'texts']

# Explore the dataset
print('Out of {} rows, {} are spam, {} are ham'.format(len(data), len(data[data['labels']=='spam']), len(data[data['labels']=='ham'])))
# Check the Number of missing data
print('Number of null in labels: {} and number of null in texts: {}'.format(data['labels'].isnull().sum(), data['texts'].isnull().sum()))

# stopwords removal
stopwords = nltk.corpus.stopwords.words('english')

# Wordnetlemmatizer
wm = nltk.WordNetLemmatizer()
# pre-processing data
def data_clean(texts):
    text = "".join([char for char in texts if char not in string.punctuation])
    tokens = re.split('W+', text)
    text = [wm.lemmatize(word) for word in tokens if word not in stopwords]
    return text

data['cleaned_text'] = data['texts'].apply(lambda x: data_clean(x.lower()))

# Vectorizing
tfidf_vect = TfidfVectorizer(analyzer=data_clean)
X_tfidf = tfidf_vect.fit_transform(data['cleaned_text'])
import ipdb; ipdb.set_trace()
print(X_tfidf.shape, tfidf_vect.get_feature_names())




