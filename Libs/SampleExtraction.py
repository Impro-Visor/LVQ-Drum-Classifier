
# coding: utf-8

# Imports and Setup
import numpy as np
import csv, sys
import matplotlib.pyplot as plt
import librosa

print str(sys.argv)

try:
    if len(sys.argv) < 3:
        file_in   = sys.argv[1]
        print "No destination specified, setting to current directory"
        file_dest = './'
    elif len(sys.argv) < 2:
        print "Need to specify 2 arguments, WAV file and destination of onset CSV file"
        sys.exit(2)
    else:
        file_in, file_dest = sys.argv[1], sys.argv[2]        

    y, sr = librosa.load(file_in)
    S     = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    o_env = librosa.onset.onset_strength(y, sr=sr)
    log_S = librosa.logamplitude(S, ref_power=np.max)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    D = np.abs(librosa.stft(y))**2
    
    onset_stfts = []
    for frame in onset_frames:
        print np.abs(D[:, frame])
        onset_stfts.append(np.abs(D[:, frame]))

    filename = raw_input('Enter filename for %s' % file_in)
    with open(file_dest + filename, 'w+') as f:
        fw = csv.writer(f, delimiter=',')
    
        for i in range(len(onset_frames)):
            # onset_stfts is an array of ndarrays so we need to cast to list
            fw.writerow([onset_frames[i], onset_stfts[i].tolist()]) 
except:
    print "Error occurred!"
    sys.exit(2)
