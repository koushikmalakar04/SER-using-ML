#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os
import seaborn as sns
import matplotlib.pyplot as plt 
import librosa 
import librosa.display
from IPython.display import Audio
import warnings 
warnings.filterwarnings('ignore')
print("Module Imported")


# In[2]:


paths = []
labels = []
for dirname, _, filenames in os.walk('F:/TESS/TESS Toronto emotional speech set data'):
     for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())

print('Dataset is loaded')


# In[3]:


paths[:5]


# In[4]:


labels[:5]


# In[5]:


df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


# In[6]:


df['label'].value_counts()


# In[7]:


sns.countplot(df['label'])


# Feature Extraction 

# In[8]:


def extract_mfcc(filename):
    y, sr=librosa.load(filename,duration=3,offset=0.5) 
    mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc 


# In[9]:


extract_mfcc(df['speech'][0])


# In[10]:


X_mfcc=df['speech'].apply(lambda x:extract_mfcc(x))


# In[11]:


X_mfcc


# In[12]:


X=[x for x in X_mfcc]
X=np.array(X)
X.shape


# Input Split

# In[13]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])


# In[14]:


y = y.toarray()


# In[15]:


y.shape


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y)


# In[17]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[18]:


from keras.layers import Dropout,Dense,TimeDistributed
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[19]:


import time
t0 = time.time()
model.fit(x_train,y_train)
print("Training Time: ", time.time()-t0)


# In[20]:


import pickle
Pkl_Filename = "Emotion_Voice_Detection_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# In[21]:


with open(Pkl_Filename, 'rb') as file:  
    Emotion_Voice_Detection_Model = pickle.load(file)

Emotion_Voice_Detection_Model


# In[22]:


y_pred=Emotion_Voice_Detection_Model.predict(x_test)
y_pred


# In[23]:


from sklearn.metrics import confusion_matrix


# In[24]:


con_matrix=pd.DataFrame(confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)),columns = list(range(0,7)))
print(con_matrix)


# In[25]:


diag = np.diagonal(con_matrix)
print(diag)


# In[26]:


sum1=np.sum(con_matrix)


# In[27]:


sum2=np.sum(sum1)


# In[28]:


diag_sum=np.sum(diag)


# In[29]:


accuracy= diag_sum/sum2
print(accuracy)


# In[30]:


import pyaudio
import wave

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output10.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# In[31]:


data, sampling_rate = librosa.load('output10.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import librosa.display
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveshow(data, sr=sampling_rate)


# In[32]:


from pydub import AudioSegment
from pydub.playback import play
 
# for playing mp3 file
song = AudioSegment.from_mp3("output10.wav")
print('playing sound using  pydub')
play(song)


# In[33]:


file = 'output10.wav'
data , sr = librosa.load(file)
data = np.array(data)
ans =[]
new_feature  = extract_mfcc(file)
print(new_feature.shape)
ans.append(new_feature)
ans = np.array(ans)
print(ans.shape)
data.shape

Emotion_Voice_Detection_Model.predict(ans)

