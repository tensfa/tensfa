X = []
sentences = list(titles["title"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = titles['Unnamed: 1']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1 #vocab_size 43

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array
from numpy import asarray
from numpy import zeros

from gensim.models import KeyedVectors
embeddings_dictionary = KeyedVectors.load_word2vec_format('drive/My Drive/trmodel', binary=True)

from keras.layers.recurrent import LSTM

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embeddings_dictionary.vectors], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])