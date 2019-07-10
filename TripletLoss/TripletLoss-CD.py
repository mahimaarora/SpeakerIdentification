
# IMPORTING LIBRARIES
import librosa
import decimal
import math
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Reshape,Flatten,Dense,add
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation
from keras.models import Model, Sequential, load_model
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
import csv
import random
import copy
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter, butter
import matplotlib.pyplot as plt


# LOADING TRAIN DATA

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

#details.npy contains the labels for speakers and number of speakers
speakers, n = np.load('data/details.npy',allow_pickle=True)

#training samples contains audio path and label for that file
training_samples = np.load('data/training_samples.npy', allow_pickle=True).tolist()
samples_number = [len(i) for i in training_samples]
total_samples = sum(samples_number)


# TRIPLETS FORMATION

"""Training Samples for each speaker: 15
Each sample is taken as an anchor and random selection for positive and negative samples """

# Returns the triplets formed from the training samples
def generate_triplets(data):
    data_copy_1 = copy.deepcopy(data)
    data_copy = copy.deepcopy(data)
    triplets = []
    for speaker in range(n):
        for audio in data_copy_1[speaker]:
            data_copy = copy.deepcopy(data)
            triplet = [audio]
            data_copy[speaker].remove(audio)
            random.shuffle(data_copy[speaker])
            triplet.append(data_copy[speaker].pop(0))
            del data_copy[speaker]
            negative_id = random.randint(0, len(data_copy)-1)
            random.shuffle(data_copy[negative_id])
            triplet.append(data_copy[negative_id].pop(0))
            triplets.append(triplet)
        
    return triplets
    
label_data = copy.deepcopy(training_samples)
triplets_data = generate_triplets(label_data)


# Verifying the generate_triplets function
check = 0
for i in triplets_data:
    if (i[0][1] == i[1][1]) and (i[0][1] != i[2][1]):
        check +=1
print(total_samples == len(triplets_data) == check)


# CONSTANTS
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025    # 25ms
FRAME_STEP = 0.01    # 10ms
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 3
WEIGHTS_FILE = "data/SavedModel/weights.h5"
COST_METRIC = "cosine" 
INPUT_SHAPE=(NUM_FFT,300,1)    #For 3 seconds (fixed input size)
ROUNDS=5


# REPRESENTATION OF AUDIO DATA (SPEECH TO SPECTRUM)

    """https://github.com/jameslyons/python_speech_features
    This file includes routines for basic signal processing including framing and computing power spectra.
    Author: James Lyons 2012"""

# Reads the input file
def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()    
    return audio

"""https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m"""
# Using lfilter to remove the noise from data
def remove_dc_and_dither(sin, sample_rate):
    alpha = 0.99
    sin = lfilter([1,-1], [1,-alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout

# Used for noise reduction (boosting of higher frequencies)
def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# Used as a sliding window while framing
def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

# Divides the signal into frames
def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):    
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    win = winfunc(frame_len)
    frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    
    return frames * win


def normalize_frames(m,epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

# returns a dictionary to fix size of spectrum based on the audio length
def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1    # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1    # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1    # mpool5
        s = np.floor((s-1)/1) + 1    # fc6
        s = np.floor((s-1)/1) + 1    # fc7
        s = np.floor((s-1)/1) + 1    # fc8
        s = np.floor((s-1)/1) + 1    # fc9
        if s > 0:
            buckets[i] = int(s)
    return buckets
buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)

# Returns the FFT spectrum for given audio file ( 512 x 300 for 3 seconds)
def get_fft_spectrum(filename):
    
    signal = load_wav(filename,SAMPLE_RATE)
    signal *= 2**15

    # Get FFT spectrum
    signal = remove_dc_and_dither(signal, SAMPLE_RATE)
    signal = preemphasis(signal, coeff=PREEMPHASIS_ALPHA)
    frames = framesig(signal, frame_len=FRAME_LEN*SAMPLE_RATE, frame_step=FRAME_STEP*SAMPLE_RATE, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=NUM_FFT))
    fft_norm = normalize_frames(fft.T)

    # Truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:,rstart:rstart+rsize]

    return out


# PRE-TRAINED MODEL

def VGGModel(INP):

    # Layer 1
    x = ZeroPadding2D(padding=(1,1))(INP)
    x = Conv2D(filters=96,kernel_size=(7,7), strides=(2,2), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    # Layer 2
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(filters=256,kernel_size=(5,5), strides=(2,2), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    # Layer 3
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)

    # Layer 4
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(filters=256,kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)

    # Layer 5
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(filters=256,kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)

    # Layer 6
    x = ZeroPadding2D(padding=(0,0))(x)
    x = Conv2D(filters=4096,kernel_size=(9,1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,4096))(x)

    # Layer 7
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5,momentum=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=None)(x)

    # Layer 8
    x = Lambda(lambda y: K.l2_normalize(y, axis=3))(x)
    x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    
    MODEL = Model(INP, x, name='VGGModel')
    return MODEL


# TRIPLET LOSS MODEL    

# INPUT: FFT Spectrum, OUTPUT: 1024 Dimentional Feature Vector
def encoder(inp):

    # Transfer Learning    
    base_model = VGGModel(inp)
    base_model.load_weights(WEIGHTS_FILE)

    #Popping out last 8 layers from the pre-trained model
    poppedModel = Model(base_model.input,base_model.layers[-8].output)
        # for i,layer in enumerate(poppedModel.layers):
        #     print(i,layer.name)
    for layer in poppedModel.layers:
        layer.trainable=False

    # Adding Trainable Layers

    x = Conv2D(filters=2048,kernel_size=(1,1), strides=(1,1), padding='valid')(poppedModel.layers[-1].output)
    x = Conv2D(filters=2048,kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=3))(x)
    x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid')(x)

    return x


# Taking ANCHOR, POSITIVE and NEGATIVE as input and returning the encoded vector representation
def build_network(encoder,input_shape_1,input_shape_2,input_shape_3):    
    input_1 = Input(input_shape_1)
    input_2 = Input(input_shape_2)
    input_3 = Input(input_shape_3)
    
    anchor = encoder(input_1)
    positive = encoder(input_2)
    negative = encoder(input_3)
    print('Encoder built!')
    
    model_ = Model(inputs=[input_1, input_2,input_3], outputs=[anchor,positive,negative])

    return model_


"""The loss function is defined as:
                                L(a,p,n) = max(d(a,p) - d(a,n) + margin,0)
Distance Metric used: Cosine Distance"""

def triplet_loss(y_true,y_pred):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    def cosine_distance(vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def cos_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0],1)

    pos_dist = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, positive])
    neg_dist = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([anchor, negative])

    loss = tf.reduce_sum(pos_dist - neg_dist)
    return loss  

triplet_loss_model = build_network(encoder,INPUT_SHAPE,INPUT_SHAPE,INPUT_SHAPE)


#COMPILATION OF THE TRIPLET LOSS MODEL
from keras.optimizers import Adam
opt = Adam(lr=0.0001)
triplet_loss_model.compile(optimizer=opt,loss=triplet_loss)


# PREPARING TRAINING DATA

# Finding spectrum(embedding) for each training sample
embeddings_df = pd.read_csv('data/train.csv')
embeddings_df['Embedding'] = embeddings_df['SampleName'].apply(lambda x: get_fft_spectrum(x))

train_header = ['ID_anchor','emb_anchor','ID_positive','emb_positive','ID_negative','emb_negative']

# Writing the triplets formed in a file
def get_audio_embeddings(triplets):
    with open('data/train_triplet_loss.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_header)
        writer.writeheader()
        for triplet in triplets:
            row = []
            row.append(triplet[0][1])
            row.append(triplet[0][0])
            row.append(triplet[1][1])
            row.append(triplet[1][0])
            row.append(triplet[2][1])
            row.append(triplet[2][0])            
            writer.writerow(dict(zip(train_header, row)))
            
    train_df = pd.read_csv('data/train_triplet_loss.csv')    
    return train_df['emb_anchor'],train_df['emb_positive'],train_df['emb_negative']

X_A,X_P,X_N = get_audio_embeddings(triplets_data)
train_df = pd.read_csv('data/train_triplet_loss.csv') 

# Output y for model randomly created as not requiredwhile calculating loss
triplet_len = len(triplets_data)
Y = [np.random.rand(1024,).reshape(-1,1024,) for i in range(triplet_len)]
y = [np.array(Y),np.array(Y),np.array(Y)]
# Mapping value of each triplet to its spectrum values
X_A_emb = list(train_df['emb_anchor'].map(embeddings_df.set_index('SampleName')['Embedding']))
X_P_emb = list(train_df['emb_positive'].map(embeddings_df.set_index('SampleName')['Embedding']))
X_N_emb = list(train_df['emb_negative'].map(embeddings_df.set_index('SampleName')['Embedding']))

# Training the model and saving the model along with weights
triplet_loss_model.fit([np.array(X_A_emb).reshape(-1,512,300,1), np.array(X_P_emb).reshape(-1,512,300,1), np.array(X_N_emb).reshape(-1,512,300,1)],[y[0],y[1],y[2]],epochs=300,batch_size=30)
triplet_loss_model.save('Results/model_tl.h5')
triplet_loss_model.save_weights('Results/weights_tl.h5')


#  VALIDATION AND TESTING

# To store validation accuracy and testing accuracy results for mulitple rounds
val_acc = []
test_acc = []

# Addition of more triplets to training data based on validation round
def recreate_training_data(speaker, prediction):
    data_val = embeddings_df[['SampleName','Speaker','Embedding']]
    size = len(speaker)
    A_VAL = []
    P_VAL = []
    N_VAL = []
    for i in range(size):
        if speaker[i] != prediction[i]:
            for _ in range(1):
                [a,p] = data_val[data_val['Speaker'] == speaker[i]]['Embedding'].sample(2).tolist()
                [n] = data_val[data_val['Speaker'] == prediction[i]]['Embedding'].sample(1).tolist()
                A_VAL.append(a)
                P_VAL.append(p)
                N_VAL.append(n)                 
                
    return A_VAL, P_VAL, N_VAL

# Converts training emeddings to corresponding vectors
def convert_emb_to_vec_train(model, embeddings_df):
    embeddings_df['Vector'] = embeddings_df['Embedding'].apply(lambda x: np.squeeze(model.predict(np.array(x).reshape(1,512,300,1))))
    return embeddings_df[['Speaker','Vector']]

# Converts validation and testing emeddings to corresponding vectors
def convert_emb_to_vec(model, df):
    df['Embedding'] = df['SampleName'].apply(lambda x: get_fft_spectrum(x))
    df['Vector'] = df['Embedding'].apply(lambda x: np.squeeze(model.predict(np.array(x).reshape(1,*np.array(x).shape,1))))
    return df[['SampleName','Speaker','Vector']]


"""Validation Samples: 5
   Testing Samples: 5
   Model tested on validation samples. 
   For each incorrect prediction one triplet is added to the training set of triplets.
   Model is retrained, saved and tested on Test Samples."""  
   
for roundno in range(ROUNDS):
    print('VALIDATION AND TESTING ROUND',roundno+1,'STARTED!')

    print('Loading trained Model....')
    try_model = load_model('Results/model_tl.h5', custom_objects={'triplet_loss':triplet_loss})
    try_model.load_weights('Results/weights_tl.h5')

    print('Extracting encoder from trained Model...')
    encoder_trained = Model(try_model.layers[0].input,try_model.layers[-3].output)

    print('Processing train samples...')
    train_result = convert_emb_to_vec_train(encoder_trained, embeddings_df)
    # print('train_result', train_result)
    train_final = train_result.groupby('Speaker')['Vector'].apply(np.mean)
    train_final = train_final.reset_index()
    # print('train_final', train_final)
    train_final.columns = ['Speaker', 'Vector']
    trained_vec = np.array([vec.tolist() for vec in train_final['Vector']])
    speakers = train_final['Speaker']

    print('\nProcessing Validation samples...')
    val_df = pd.read_csv('data/validation.csv')
    val_result = convert_emb_to_vec(encoder_trained, val_df)
    val_vec= np.array([vec.tolist() for vec in val_result['Vector']])

    print("Comparing validation samples against trained samples....")
    distances = pd.DataFrame(cdist(val_vec, trained_vec, metric=COST_METRIC), columns=speakers)
    val_df = pd.concat([val_df, distances],axis=1)
    val_df['PredictedSpeaker'] = val_df[speakers].idxmin(axis=1)
    val_df['Correct'] = (val_df['PredictedSpeaker'] == val_df['Speaker'])*1. # bool to int
    print('\nNumber of validation samples: ', val_df.shape[0])
    print('Number of correctly predicted speakers: ', sum(val_df['Correct'].tolist()))
    validation_accuracy = sum(val_df['Correct'].tolist())/ val_df.shape[0]
    print('Validation Accuracy:',validation_accuracy)
    val_acc.append(validation_accuracy)

    
    print('\nAdding triplets...')
    result = val_df[['SampleName','Speaker','PredictedSpeaker', 'Correct']]
    speaker_val = result['Speaker'].tolist()
    prediction_val = result['PredictedSpeaker'].tolist()

    Anchor,Positive,Negative= recreate_training_data(speaker_val, prediction_val)

    X_A_emb = X_A_emb + Anchor
    X_P_emb = X_P_emb + Positive
    X_N_emb = X_N_emb + Negative
    data_size = len(X_A_emb)
    Y = [np.random.rand(1024,).reshape(-1,1024,) for i in range(data_size)]
    y = [np.array(Y),np.array(Y),np.array(Y)]
    batch_size = 40
    while data_size % batch_size <15:
        batch_size+=5
  
    print('DataSize:',data_size,'BatchSize:',batch_size)
    print('Retraining the model...')
    try_model.fit([np.array(X_A_emb).reshape(-1,512,300,1), np.array(X_P_emb).reshape(-1,512,300,1), np.array(X_N_emb).reshape(-1,512,300,1)],[y[0],y[1],y[2]],epochs=60,batch_size=batch_size)
    print('\nSaving the retrained Model...')
    try_model.save('Results/model_tl.h5')
    try_model.save_weights('Results/weights_tl.h5')

    
    print('Extracting encoder from trained Model...')
    encoder_trained = Model(try_model.layers[0].input,try_model.layers[-3].output)

    print('Processing train samples...')
    train_result = convert_emb_to_vec_train(encoder_trained, embeddings_df)
    train_final = train_result.groupby('Speaker')['Vector'].apply(np.mean)
    train_final = train_final.reset_index()
    train_final.columns = ['Speaker', 'Vector']
    trained_vec = np.array([vec.tolist() for vec in train_final['Vector']])
    speakers = train_final['Speaker']

    print('\nProcessing Testing samples...')
    test_df = pd.read_csv('data/test.csv')
    test_result = convert_emb_to_vec(encoder_trained, test_df)
    test_vec= np.array([vec.tolist() for vec in test_result['Vector']])

    print("Comparing test samples against trained samples....")
    distances = pd.DataFrame(cdist(test_vec, trained_vec, metric=COST_METRIC), columns=speakers)
    test_df = pd.concat([test_df, distances],axis=1)
    test_df['PredictedSpeaker'] = test_df[speakers].idxmin(axis=1)
    test_df['Correct'] = (test_df['PredictedSpeaker'] == test_df['Speaker'])*1. # bool to int
    print('\nNumber of test samples: ', test_df.shape[0])
    print('Number of correctly predicted speakers: ', sum(test_df['Correct'].tolist()))
    test_accuracy = sum(test_df['Correct'].tolist())/ test_df.shape[0]
    print('Test Accuracy:',test_accuracy)
    test_acc.append(test_accuracy)
    results = test_df[['SampleName','Speaker', 'PredictedSpeaker', 'Correct']]
    RESULTS = results.to_csv('results.csv')
    print('VALIDATION AND TESTING ROUND',roundno+1,'COMPLETED!\n\n')


# PLOTTING THE VALIDATION AND TEST ACCURACY

x_axis = [i for i in range(len(test_acc))]
labels = ['Test Accuracy', 'Validation Accuracy']
plt.plot(test_acc, marker='o',color='red')
plt.plot(val_acc, marker='o')
plt.ylim(top=1,bottom=0)
plt.xlabel('Iteration Number')
plt.ylabel('Accuracy Score')
plt.title('Validation and Test Accuracy')
plt.legend(labels)
plt.xticks(x_axis)
for i,j in zip(x_axis,test_acc):
    plt.annotate(str(j),xy=(i,j))
plt.show()

# Saving the accuracy results
acc_details=[test_acc,val_acc]
np.save('data/acc_details.npy', acc_details)


# Loading the saved model with weights
print('Loading trained Model....')
try_model = load_model('Results/model_tl.h5', custom_objects={'triplet_loss':triplet_loss})
try_model.load_weights('Results/weights_tl.h5')
print('Extracting encoder from trained Model...')
encoder_trained = Model(try_model.layers[0].input,try_model.layers[-3].output)


# ADDING NEW SPEAKER
print('Reading New Data...')
newdata = pd.read_csv('data/newdata.csv')
newdata['Embedding'] = newdata['SampleName'].apply(lambda x: get_fft_spectrum(x))
newdata['Vector'] = newdata['Embedding'].apply(lambda x: np.squeeze(encoder_trained.predict(np.array(x).reshape(1,512,300,1))))

print('Adding New Data...')
newemb = [embeddings_df, newdata]
new_embeddings_df = pd.concat(newemb)
new_embeddings_df.shape[0] == embeddings_df.shape[0] + newdata.shape[0]


# TESTING FOR A SINGLE AUDIO SAMPLE

def test_sample(sample, embeddings_df):

    embedding = get_fft_spectrum(sample)
    vector = np.squeeze(encoder_trained.predict(np.array(embedding).reshape(1,512,300,1)))
    
    embeddings_df['distance'] = embeddings_df['Vector'].apply(lambda x: scipy.spatial.distance.cosine(x, vector))
    sample_id = embeddings_df.loc[embeddings_df['distance'].idxmin()]
    print(embeddings_df['distance'].describe())
    embeddings_df = embeddings_df.drop(columns=['distance'])
    print(sample_id['Speaker'])
    return embeddings_df

embeddings_df = test_sample('data_test/id10004.10.wav', embeddings_df)
