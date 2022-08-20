#importing all the library
import tensorflow as tf;
from tensorflow import keras;
import numpy as np;


#imdb dataset present in keras for movie reviews
imdb=keras.datasets.imdb;
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000);


#the raw traing data collected from the module
'''
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0]);
len(train_data[0]), len(train_data[1])
'''

#print(train_data[0]);
#A dictionary mapping words to an integer index
word_index = imdb.get_word_index()



# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3


#dictionary mapping from word to value
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



#decoding the integer stream to words
def decode_review(text):
	return(' '.join([reverse_word_index.get(i,'?') for i in text]))
print(decode_review(train_data[0]))
print()
print("the test label is ")
print(train_labels[0]);	





#fixing the size of each comment to 256 length by padding or trumcating
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])




#building the neural network model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()



#TUNING THE HYPERPARAMETERS LIKE LEARNING RATE AND ALGORITHMS 


#ALL THE ALGORITHMS THAT ARE USED FOR ACCURACY 

#STOCHASTIC GRADIENT DISCENT
'''
sgd=keras.optimizers.SGD(learning_rate=0.005);
model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['acc']);
'''

#RMSprop
'''
rmsprop=keras.optimizers.RMSprop(learning_rate=0.005);
model.compile(optimizer=rmsprop,loss='mean_squared_error',metrics=['acc']);
'''

#Adagrad
'''
adagrad=keras.optimizers.Adagrad(learning_rate=0.005)#nadam=keras.optimizers.Nadam(learning_rate=0.005)
model.compile(optimizer=adagrad,loss='mean_squared_error',metrics=['acc']);
'''


#Adam(Optimized Algorithm)
adam=keras.optimizers.Adam(learning_rate=0.005);
model.compile(optimizer=adam,loss='mean_squared_error',metrics=['acc']);


#Nadam
'''
nadam=keras.optimizers.Nadam(learning_rate=0.005);
model.compile(optimizer=nadam,loss='mean_squared_error',metrics=['acc']);
'''

#Dividing the training data into traning set and validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]





#training the model
history=model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val, y_val),verbose=1);
#testing the model
results = model.evaluate(test_data, test_labels)
print(results);	




#FINAL RESULTS
#--------------


#(i)getting the sentiment of the FIRST single comment...
test_review=test_data[1]
predict=model.predict(test_review);
print("review is ");
print(decode_review(test_review));
print("prediction: is "+str(predict[1]));
print("actual is : "+str(test_labels[1]));


#(ii)getting the sentiment of the SECOND single comment...
test_review=test_data[2]
predict=model.predict(test_review);
print("review is ");
print(decode_review(test_review));
print("prediction: is "+str(predict[2]));
print("actual is : "+str(test_labels[2]));


#(iii)getting the sentiment of the THIRD single comment...
test_review=test_data[3]
predict=model.predict(test_review);
print("review is ");
print(decode_review(test_review));
print("prediction: is "+str(predict[3]));
print("actual is : "+str(test_labels[3]));
