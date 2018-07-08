import numpy as np
import os
import sys
import datetime
import time
import re
import pandas as pd
import keras
import sklearn

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation
from keras.optimizers import Nadam
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

def clean_str(text):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r" e g ", " eg ", text)
	text = re.sub(r" b g ", " bg ", text)
	text = re.sub(r" u s ", " american ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)

	return text.strip().lower()
	
def load_train_data_and_labels(src_csv):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	df = pd.read_csv(src_csv, sep=',')
	df.dropna(inplace=True)
	name = df['name']
	desc = df['desc']
	goal = df['goal']
	country	= df['country']
	currency = df['currency']
	deadline = df['deadline']
	launched_at = df['launched_at']
	backers = df['backers_count']
	stat = df['final_status']

	x_desc = [s.strip() for s in desc]
	x_name = [s.strip() for s in name]
	# Split by words
	x_desc = [clean_str(sent) for sent in x_desc]
	x_name = [clean_str(sent) for sent in x_name]
	
	x_feat = []
	for i in range(len(stat)):
		x_feat.append([goal.iloc[i], 
				country.iloc[i], 
				currency.iloc[i], 
				deadline.iloc[i], 
				launched_at.iloc[i],
				backers.iloc[i]])

	return [x_name, 
			x_desc, 
			np.array(x_feat),
			stat]


DATA_DIR = 'data/'
MODEL_DIR = 'model/'
EMBEDDING_FILE = DATA_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = DATA_DIR + 'train.csv'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 100
num_dense = 50
rate_drop_lstm = 0.45
rate_drop_dense = 0.25

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

print('Processing text dataset')

x_name, x_desc, x_feat, y = load_train_data_and_labels('data/train.csv')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_name + x_desc)

#Save tokenizer
TOKENIZER_FILE = MODEL_DIR + STAMP + 'tokenizer.pkl'
joblib.dump(tokenizer, TOKENIZER_FILE)

sequences_names = tokenizer.texts_to_sequences(x_name)
sequences_desc = tokenizer.texts_to_sequences(x_desc)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_names = pad_sequences(sequences_names, maxlen=MAX_SEQUENCE_LENGTH)
data_desc = pad_sequences(sequences_desc, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(y)
print('Shape of data tensor:', data_names.shape)
print('Shape of res tensor:', y.shape)


print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

word2vec = None
sequences_desc = None
sequences_names = None
x_name = None
x_desc = None

print('Preparing features')

label_encoder_country = LabelEncoder()
x_feat[:, 1] = label_encoder_country.fit_transform(x_feat[:, 1])

label_encoder_currency = LabelEncoder()
x_feat[:, 2] = label_encoder_currency.fit_transform(x_feat[:, 2])
x_feat = x_feat.astype(float)
for i in range(len(x_feat)):
	x_feat[i] = x_feat[i].astype(float)

index = 1
print('Features: ', x_feat[index,:])
print('Shape of features: ', x_feat.shape)


#Save encoders
COUNTRY_ENCODER_FILE = MODEL_DIR + STAMP + 'country.pkl'
joblib.dump(label_encoder_country, COUNTRY_ENCODER_FILE)
CURRENCY_ENCODER_FILE = MODEL_DIR + STAMP + 'currency.pkl'
joblib.dump(label_encoder_currency, CURRENCY_ENCODER_FILE)


data_names_train, data_names_test, data_desc_train, data_desc_test, \
	data_feat_train, data_feat_test, y_train, y_test = \
	train_test_split(data_names, data_desc, x_feat, y, test_size=0.2)
	
data_names_train, data_names_val, data_desc_train, data_desc_val, data_feat_train, data_feat_val, y_train, y_val = \
	train_test_split(data_names_train, data_desc_train, data_feat_train, y_train, test_size=VALIDATION_SPLIT)
	
data_names, data_desc, x_feat, y = None, None, None, None

weight_val = np.ones(len(y_test))
if re_weight:
    weight_val *= 0.472001959
    weight_val[y_test==0] = 1.309028344


embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
nameLayer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)(embedded_sequences_1)
nameLayer = BatchNormalization()(nameLayer)
nameLayer = Dense(num_dense, activation=act)(nameLayer)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
descLayer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)(embedded_sequences_2)
descLayer = BatchNormalization()(descLayer)
descLayer = Dense(num_dense, activation=act)(descLayer)

features_input = Input(shape=(data_feat_train.shape[1],))
featLayer = BatchNormalization()(features_input)
featLayer = Dense(num_dense*2, activation=act)(featLayer)
featLayer = Dropout(rate_drop_dense)(featLayer)
featLayer = Dense(num_dense, activation=act)(featLayer)

merged = concatenate([nameLayer, descLayer, featLayer])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(int(num_dense), activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)

preds = Dense(1, activation='sigmoid')(merged)

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None
	
	
opt = Nadam (lr=0.3, beta_1=0.9, beta_2=0.999, epsilon=None)
	
model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

print(STAMP)

plot_model(model, to_file=MODEL_DIR + STAMP +'model.png', show_shapes = True)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = MODEL_DIR + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)



hist = model.fit([data_names_train, data_desc_train, data_feat_train], y_train, \
        validation_data=([data_names_val, data_desc_val, data_feat_val], y_val), \
        epochs=200, batch_size=512, shuffle=True, \
        callbacks=[early_stopping, model_checkpoint])
		
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model = load_model(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

print('Start making the submission before fine-tuning')

y_pred = model.predict([data_names_test, data_desc_test, data_feat_test], batch_size=512, verbose=1)
y_classes = y_pred.flatten().astype(int)
metrics = accuracy_score(y_test, y_classes)
print('Accuracy: ' + str(metrics*100) + '%')
