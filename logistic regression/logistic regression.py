
#320937741
import numpy as np
import librosa

#same song diffrent covers
y1, sr1 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\1.mp3",duration=5)
songA_1 = librosa.feature.mfcc(y=y1,sr=sr1)
y2, sr2 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\2.mp3",duration=5)
songA_2 = librosa.feature.mfcc(y=y2, sr=sr2)

y3, sr3 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\3.mp3",duration=5)
songA_3 = librosa.feature.mfcc(y=y3, sr=sr3)
y4, sr4 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\4.mp3",duration=5)
songA_4 = librosa.feature.mfcc(y=y4, sr=sr4)
#diffrent songs
y5, sr5 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\5.mp3",duration=5)
songB = librosa.feature.mfcc(y=y5, sr=sr5)
y6, sr6 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\6.mp3",duration=5)
songC = librosa.feature.mfcc(y=y6, sr=sr6)
y7, sr7 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\7.mp3",duration=5)
songD = librosa.feature.mfcc(y=y7, sr=sr7)
y8, sr8 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\8.mp3",duration=5)
songE = librosa.feature.mfcc(y=y8, sr=sr8)

data_x = np.reshape(np.array([songA_1,songA_2,songA_3,songA_4,songB,songC,songD,songE]), (8,4320))
data_y = np.array([1,1,1,1,0,0,0,0])# 1- song A , 0- not song A

def h(x,w,b):
    return 1 / (1+np.exp(-(np.dot(x,w) + b)))

w = np.array([0.]*4320)
b = 0
alpha = 0.001

for iteration in range(100000):
    gradient_b = np.mean(1*(data_y-(h(data_x,w,b))))
    gradient_w = np.dot((data_y-h(data_x,w,b)), data_x)*1/len(data_y)
    b += alpha*gradient_b
    w += alpha*gradient_w


	
y1, sr1 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\1.mp3",duration=5)	
song_1 = np.reshape(librosa.feature.mfcc(y=y11, sr=sr1),(4320,))
print("the propability that this song is song A is: ", h((song_1),w,b))

y2, sr2 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\2.mp3",duration=5)	
song_2 = np.reshape(librosa.feature.mfcc(y=y2, sr=sr2),(4320,))
print("the propability that this song is song A is: ", h((song_2),w,b))

y3, sr3 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\3.mp3",duration=5)	
song_3 = np.reshape(librosa.feature.mfcc(y=y3, sr=sr3),(4320,))
print("the propability that this song is song A is: ", h((song_3),w,b))


y4, sr4 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\4.mp3",duration=5)	
song_4 = np.reshape(librosa.feature.mfcc(y=y4, sr=sr4),(4320,))
print("the propability that this song is song A is: ", h((song_4),w,b))

y5, sr5 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\5.mp3",duration=5)	
song_5 = np.reshape(librosa.feature.mfcc(y=y5, sr=sr5),(4320,))
print("the propability that this song is song A is: ", h((song_9),w,b))


y6, sr6 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\6.mp3",duration=5)	
song_6 = np.reshape(librosa.feature.mfcc(y=y6, sr=sr6),(4320,))
print("the propability that this song is song A is: ", h((song_6),w,b))


y7, sr7 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\7.mp3",duration=5)	
song_7 = np.reshape(librosa.feature.mfcc(y=y7, sr=sr7),(4320,))
print("the propability that this song is song A is: ", h((song_7),w,b))

y8, sr8 = librosa.load(r"C:\Users\rita1\AppData\Local\Programs\Python\Python36\data\Train\8.mp3",duration=5)	
song_8 = np.reshape(librosa.feature.mfcc(y=y8, sr=sr8),(4320,))
print("the propability that this song is song A is: ", h((song_8),w,b))

