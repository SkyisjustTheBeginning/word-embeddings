import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
words = ['run','learn','fight','stand','fast','small','big','slow']
label = np.array([0,0,0,0,1,1,1,1])
vocab_size = 50
encoded_docs = [one_hot(w, vocab_size) for w in words]
mlength = 6
padded_docs = pad_sequences(encoded_docs, maxlen=mlength, padding='post')
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=mlength))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_docs, label, epochs=50, verbose=0)
loss, accuracy = model.evaluate(padded_docs, label, verbose=0)
print("Loss :  " + str(int(loss *100)))
print("Accuracy :  " + str(int(accuracy *100)))
