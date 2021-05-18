import os
import spacy
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import pickle

def load_data():
    if not os.path.exists('data/train_data.xlsx'):
        stack_overflow = pd.read_excel('../SFData/StackOverflow.xlsx')
        stack_overflow = stack_overflow[['question id', 'exception type', 'exception content']]
        stack_overflow['is shape fault'] = [1]*len(stack_overflow)
        stack_overflow['from'] = ['stack_overflow']*len(stack_overflow)

        ICSE2020 = pd.read_excel('../SFData/ICSE2020ToDetect.xlsx')
        ICSE2020 = ICSE2020[['question id', 'exception type', 'exception content', 'is shape fault']]
        ICSE2020['from'] = ['ICSE2020']*len(ICSE2020)

        ISSTA2018 = pd.read_excel('../SFData/ISSTA2018NoTensorShapeFault.xlsx')
        ISSTA2018 = ISSTA2018[['question id', 'exception type', 'exception content']]
        ISSTA2018['is shape fault'] = [0] * len(ISSTA2018)
        ISSTA2018['from'] = ['ISSTA2018'] * len(ISSTA2018)

        keras_exception = pd.read_excel('../SFData/KerasException.xlsx')
        keras_exception['is shape fault'] = [0] * len(keras_exception)
        keras_exception['from'] = ['keras exception'] * len(keras_exception)

        data = pd.concat([stack_overflow, ICSE2020, keras_exception], axis=0,  ignore_index=True)
        data = data.sample(frac=1, random_state=2021) # shuffle

        test_ids = pd.read_excel('../SFData/ICSE2020ToRepair.xlsx')['question id'].to_list()
        test_ids = set(test_ids)
        is_test_id = data['question id'].apply(lambda i: i in test_ids)
        train_data = data[~is_test_id]

        test_data = data[is_test_id]
        test_data = pd.concat([test_data, ISSTA2018], axis=0, ignore_index=True)

        print(len(train_data), len(train_data[train_data['is shape fault'] == 1]))
        print(len(test_data), len(test_data[test_data['is shape fault'] == 1]))

        train_data.to_excel('data/train_data.xlsx', index=False)
        test_data.to_excel('data/test_data.xlsx', index=False)
    else:
        train_data = pd.read_excel('data/train_data.xlsx')
        test_data = pd.read_excel('data/test_data.xlsx')
    return train_data, test_data

def extract_features(train_data, test_data):
    if not os.path.exists('data/train_tfidf_features.xlsx'):
        nlp = spacy.load("en_core_web_trf")
        # tokenize
        def tokenize(string):
            doc = nlp(string)
            new_string = ' '.join([token.text for token in doc])
            return new_string
        train_data.loc[:, 'exception content'] = train_data.loc[:, 'exception content'].apply(tokenize)
        test_data.loc[:, 'exception content'] = test_data.loc[:, 'exception content'].apply(tokenize)
        vectorizer = CountVectorizer(lowercase=True, min_df=2)
        transformer = TfidfTransformer()
        train_tfidf = transformer.fit_transform(
            vectorizer.fit_transform(train_data['exception type'] + ' ' + train_data['exception content']))
        train_tfidf_features = pd.DataFrame(train_tfidf.toarray())
        test_tfidf = transformer.transform(
            vectorizer.transform(test_data['exception type'] + ' ' + test_data['exception content']))
        test_tfidf_features = pd.DataFrame(test_tfidf.toarray())
        train_tfidf_features.to_excel('data/train_tfidf_features.xlsx', index=False)
        test_tfidf_features.to_excel('data/test_tfidf_features.xlsx', index=False)
        pickle.dump(vectorizer, open('data/vectorizer', 'wb'))
        pickle.dump(transformer, open('data/transformer', 'wb'))
    else:
        train_tfidf_features = pd.read_excel('data/train_tfidf_features.xlsx')
        test_tfidf_features = pd.read_excel('data/test_tfidf_features.xlsx')
    return train_tfidf_features, test_tfidf_features

train, test = load_data()
# enc = OneHotEncoder(handle_unknown='ignore')
# train_et_features = pd.DataFrame(enc.fit_transform(train_data[['exception type']]).toarray())
# test_et_features = pd.DataFrame(enc.transform(test_data[['exception type']]).toarray())
train_features, test_features = extract_features(train, test)
vectorizer = pickle.load(open('data/vectorizer', 'rb'))
words = vectorizer.get_feature_names()
# train_features = pd.concat([train_et_features, train_tfidf_features], axis=1).values
train_labels = train['is shape fault']
# test_features = pd.concat([test_et_features, test_tfidf_features], axis=1).values
test_labels = test['is shape fault']

tree = DecisionTreeClassifier(random_state=2021, max_depth=5)
tree.fit(train_features, train_labels)
pickle.dump(tree, open('data/tree', 'wb'))

print('===================train===================')
predicted_lables = tree.predict(train_features)
print(classification_report(train_labels, predicted_lables))

predicted_lables = tree.predict(test_features)
print('===================test===================')
print(predicted_lables)
print(classification_report(test_labels, predicted_lables))
