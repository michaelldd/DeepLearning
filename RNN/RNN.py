from os import listdir
from sklearn.preprocessing import normalize
import numpy as np
from numpy.random import shuffle
import librosa
import tensorflow as tf

def GetDefaultParameters():

    Data = {
        'trans_data': "/Users/rita1/PycharmProjects/SongClassification/mfcc/trans_data",
        'raw_data': "/Users/rita1/PycharmProjects/SongClassification/mfcc/raw_data",
        'test_data': "/Users/rita1/PycharmProjects/SongClassification/mfcc/test_data",
        'transTest_data': "/Users/rita1/PycharmProjects/SongClassification/mfcc/transTest_data"
    }

    Split_Ratio = {
        'train': 0.7,
        'test': 0.3
    }

    return {
        'Data': Data,
        'Split_Ratio': Split_Ratio
    }


def GetData(params):

    try:
        print("try to load .npy files")
        data, labels = get_transformed_data(params['trans_data'])
    except:
        print('The files does not exist. Loading the songs and save as .npy files')
        data, labels = transform(params['raw_data'], params['trans_data'])
    print()
    return data, labels

def get_transformed_data(trans_path):

    return np.load(trans_path + "/" + "list_of_lists.npy"), np.load(trans_path + "/" + "labels.npy")

def transform(raw_data, trans_data):

    classes = []
    #listdir - Return a list containing the names of the files in the directory.
    #the names of the directory are the names of the song - lables
    #all the labels placed inside the classes array
    [classes.append(item) for item in listdir(raw_data) if item.find(".DS_Store") == -1]
    features = []
    labels = []
    i=0
    # go over each version inside each song directory
    for c in classes:
        files = []
        dir_path = raw_data + "/" + c + "/"
        #all the vartions of all the songs places in the files array
        [files.append(item) for item in listdir(dir_path) if item.find(".DS_Store") == -1]
        print()
        #go over each file, load it and make featchers matrix for each of them
        for f in files:
            y, sr = librosa.load(dir_path + "/" + f)
            # display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            # Convert to log scale
            log_S = librosa.amplitude_to_db(S, ref=np.max)
            #all the featchers metrixes are placed in the features array
            mfcc = librosa.feature.mfcc(S=log_S, sr=sr)
            features.append(mfcc.tolist())
            #for each of the 20 parts of each song we place the right lable at the same index in the label array
            #so if we have 4 songs and each song has 4 versions then we'll get 16 files. each of them we
            #divide to 13 sections, so we get 16*20 = 320 sections total and each of them has a lable.
            labels.append(np.repeat(c, 20).tolist())
            i+=1
            print("loaded version {} of {}".format(f, c))

    #

    labels = [val for sublist in labels for val in sublist]
    #concatenate - Join a sequence of arrays along an existing axis
    array_of_vec = np.concatenate(np.asarray(features ), axis=0)
    #save files
    np.save(trans_data + "/" + "list_of_lists", features )
    np.save(trans_data + "/" + "labels", labels)

    return features , labels

def train_network(train_x, train_y, test_x, test_y,validation_data,validation_labels):
    x_dim = 5000
    (hidden1_size, hidden2_size) = (50, 50)#improved the results
    #every row has 5000 cells
    x = tf.placeholder(tf.float32, [None, x_dim])
    # every row has 4 cells
    y_ = tf.placeholder(tf.float32, [None, 4])
    # every row has 5000 cells
    xv = tf.placeholder(tf.float32, [None, x_dim])
    # every row has 4 cells
    yv_ = tf.placeholder(tf.float32, [None, 4])
    #weights,bieses and leaky relu
    W1 = tf.Variable(tf.truncated_normal([x_dim, hidden1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
    z1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
    z2 = tf.nn.leaky_relu(tf.matmul(z1, W2) + b2)
    W3 = tf.Variable(tf.truncated_normal([hidden2_size, 4], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[4]))
    #softmax,cross entropy and train step
    y = tf.nn.softmax(tf.matmul(z2, W3) + b3)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)#improved the results

    # graphs
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_predictionv = tf.equal(tf.argmax(y, 1), tf.argmax(yv_, 1))
    accuracyv = tf.reduce_mean(tf.cast(correct_predictionv, tf.float32))

    # declare summary
    c_e = tf.summary.scalar('cross_entropy', cross_entropy)
    a = tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    file_writer = tf.summary.FileWriter('./my_graph', sess.graph)
    sess.run(init)

    for i in range(1000):#did not improve results
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})


        if i%100==0:
            print(i, sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
            #print(i, sess.run(accuracyv, feed_dict={xv: validation_data, yv_: validation_labels}))
        summary = []
        summary = sess.run(merged, feed_dict={x: test_x, y_: test_y})
        file_writer.add_summary(summary, i)
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    #print(sess.run(accuracyv, feed_dict={xv: validation_data, yv_: validation_labels}))

if __name__ == "__main__":
    params = GetDefaultParameters()
    data, labels = GetData(params['Data'])

    # our data : 16 files each has 20 parts and each part has number of feachers - we want the parts to have the same
    # number of feachers so we cut them - we take only the 5000 firsts
    # we take those parts and put them in one row inside the X array
    X = []
    for slice in data:
        for k in slice:
            X.append(k[:5000])
    X = np.asarray(X)
    X = normalize(X)#improved the results
    print()
    # encoder labels
    encoded_labels = []
    classes = np.unique(labels)
    for l in labels:
        if l == classes[0]:
            encoded_labels.append([0.,0.,0.,1.])
        elif l == classes[1]:
            encoded_labels.append([0.,0.,1.,0.])
        elif l == classes[2]:
            encoded_labels.append([0.,1.,0.,0.])
        elif l == classes[3]:
            encoded_labels.append([1.,0.,0.,0.])

    encoded_labels = np.asarray(encoded_labels)

    # Split Train/Test
    dataset_size = X.shape[0]

    indices = np.array(range(dataset_size))
    shuffle(indices)
    train_ind = indices[:int(np.floor(dataset_size) * params['Split_Ratio']['train'])]

    test_ind = indices[int(np.floor(dataset_size) * params['Split_Ratio']['train']):]

    train_data = X[train_ind]
    test_data = X[test_ind]
    train_labels = encoded_labels[train_ind]
    test_labels = encoded_labels[test_ind]

    trainset_size = train_data.shape[0]
    trainindices = np.array(range(trainset_size))

    #shuffle(trainindices)


    #validation  - not used
    vtrain_ind = trainindices[:int(np.floor(trainset_size) * params['Split_Ratio']['train'])]
    validation_ind = trainindices[int(np.floor(trainset_size) * params['Split_Ratio']['train']):]
    vtrain_data=train_data[vtrain_ind]
    validation_data=train_data[validation_ind]
    vtrain_labels = encoded_labels[vtrain_ind]
    validation_labels=encoded_labels[validation_ind]



    train_network(train_data, train_labels, test_data, test_labels,validation_data,validation_labels)


