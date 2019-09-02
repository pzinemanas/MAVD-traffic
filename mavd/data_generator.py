import pandas as pd
import os
#import sed_eval
import gzip
import glob
import pickle
import librosa
import soundfile as sf

import numpy as np
import keras
import random
import math
from scipy.signal import hanning

from sklearn.preprocessing import StandardScaler
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for the experiments'
    def __init__(self, list_IDs, labels, scaler=[],label_list=[],train=True,
                 sequence_time=1.0, sequence_hop_time=0.5,frames=False,
                 audio_hop=882, audio_win=1764,n_fft=2048,sr=44100,mel_bands=128,
                 normalize='none',get_annotations=True,dataset='MAVD'):

        """ Initialize the DataGenerator 
        Parameters
        ----------
        list_IDs : list
            List of file IDs (i.e wav file names)
            
        labels : list
            List of file annotations (i.e txt file names)
            
        scaler : Scaler (sklearn)
            If is not [], this scaler is used 
            
        label_list : list (list of lists for MAVD)
            List of classes of interest
            If dataset == MAVD, each list is related to each level of classification
            
        train : bool
            If true, the scaler is calculated
            
        sequence_time : float
            Time in seconds of each network input (i.e 1 second length mel-spectrogram)
            
        sequence_hop_time : float
            Time in secodns of the sequence hop
            
        frames : bool
            If True the audio signal is returned in a matrix frames 
            (only useful for end-to-end networks)
            
        audio_win : int
            Number of samples of the analysis window for the STFT calculation

        audio_hop : int
            Number of samples of the hop for the STFT calculation
            
        n_fft : int
            Number of samples to calculate the FFT
            
        sr : int
            Sampling rate. If this value is different that the audio files, 
            the signals are resampled (not recomended).
            
        mel_bands : int
            Number of Mel bands.
            
        normalize : string
            'standard' to use standar normalization (sklearn)
            'minmax' to use minmax normalization
            'none' to don't normalize features
            
        get_annotations : bool
            If True, the annotations are returned
            
        dataset : 'string'
            Select the dataset 'URBAN-SED' or 'MAVD'
        
        """
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.sr = sr
        self.n_fft = n_fft
        self.mel_bands = mel_bands
        self.audio_hop = audio_hop
        self.audio_win = audio_win
        self.mel_basis = librosa.filters.mel(sr,n_fft,mel_bands,htk=True)
        self.mel_basis = self.mel_basis**2 #For end-to-end networks
        self.sequence_frames = int(sequence_time * sr / float(audio_hop))
        self.sequence_hop = int(sequence_hop_time * sr / float(audio_hop))
        self.sequence_samples = int(sequence_time * sr)
        self.sequence_hop_samples = int(sequence_hop_time * sr)
        self.hop_time = audio_hop / float(sr)
        self.label_list = label_list
        self.normalize = normalize
        self.sequence_hop_time = sequence_hop_time
        self.scaler = scaler
        self.frames = frames
        self.get_annotations = get_annotations
        self.train = train
        self.norm_scaler = np.zeros(self.mel_bands)  
        self.dataset = dataset #URBAN-SED or MAVD
        

    def __data_generation(self, list_IDs_temp):
        """ This function generates data with the files in list_IDs_temp
        
        Parameters
        ----------
        list_IDs_temp : list
            List of file IDs.

        Return
        ----------
        X : array
            Audio signals (only for end-to-end networks)
            
        S : array
            Spectrograms
            
        mel : array
            Mel-spectrograms
           
        yt : array
            Annotationes as categorical matrix

        """
        
        X = []

        if self.dataset == 'MAVD':
            yt = {}
            for index_list,l_list in enumerate(self.label_list):
                yt[index_list] = []
        if self.dataset == 'URBAN-SED':
            yt = []

        mel = []
        id_t = []
        S = []
        
        window = hanning(self.audio_win)

        for i, ID in enumerate(list_IDs_temp):
            label_file = self.labels[ID]
            
            if self.get_annotations:            
                labels = pd.read_csv(label_file, delimiter='\t', header=None)
                labels.columns = ['event_onset', 'event_offset','event_label']
            else:
                labels = pd.DataFrame({'event_onset' : []})
    
            audio,sr_old = sf.read(ID)
            if len(audio.shape) > 1:
                audio = audio[:,0]
                
            if self.sr != sr_old:
                print('changing sampling rate')
                audio = librosa.resample(audio, sr_old, self.sr)

            if self.dataset == 'MAVD':
                event_rolls = []
                for index_list,l_list in enumerate(self.label_list):
                    event_roll = np.zeros((int(math.ceil((len(audio)-self.sequence_samples+1)/ float(self.sequence_hop_samples))),
                                           len(l_list)))
                    for event in labels.to_dict('records'):
                        c1 = event['event_label']
                        c2 = c1.split('/')[0]
                        c3 = ""
                        c4 = ""
                        if len(c1.split('/')) > 1:
                            c3 = c1.split('/')[1].split('_')[0]
                            if len(c1.split('/')[0].split('_')) > 1:
                                c4 = c1.split('/')[0].split('_')[1]
                        c1 = c1.split('_')[0]
                        if (c1 in l_list) | (c2 in l_list) | (c3 in l_list) | (c4 in l_list):
                            if (c1 in l_list):
                                c = c1
                            else:
                                if (c2 in l_list):
                                    c = c2
                                else:
                                    if (c3 in l_list):
                                        c = c3
                                    else:
                                        c = c4
                            pos = l_list.index(c)

                            event_onset = event['event_onset']
                            event_offset = event['event_offset']    

                            onset = int(math.floor(event_onset * 1 / float(self.sequence_hop_time)))
                            offset = int(math.ceil(event_offset * 1 / float(self.sequence_hop_time)))

                            event_roll[onset:offset, pos] = 1
                    event_rolls.append(event_roll)
                    
            if self.dataset == 'URBAN-SED':  
                event_roll = np.zeros((int(math.ceil((len(audio)-self.sequence_samples+1)/ float(self.sequence_hop_samples))),
                                       len(self.label_list)))
                for event in labels.to_dict('records'):
                    pos = self.label_list.index(event['event_label'])
                    
                    event_onset = event['event_onset']
                    event_offset = event['event_offset']
                    
                    onset = int(math.floor(event_onset * 1 / float(self.sequence_hop_time)))
                    offset = int(math.ceil(event_offset * 1 / float(self.sequence_hop_time)))
                    
                    event_roll[onset:offset, pos] = 1
                
            for i in np.arange(0,len(audio)-self.sequence_samples+1,self.sequence_hop_samples):
                audio_slice = audio[i:i+self.sequence_samples]

                #### Normalize by slices
                if self.normalize == 'minmax':
                    audio_slice = audio[i:i+self.sequence_samples]/np.amax(audio[i:i+self.sequence_samples])
                else:
                    audio_slice = audio[i:i+self.sequence_samples]

                audio_slice[np.isinf(audio_slice)] = 0
                audio_slice[np.isnan(audio_slice)] = 0
                audio_slice_pad = np.pad(audio_slice, int(self.n_fft // 2), mode='reflect')

                if self.frames:
                    f = librosa.util.frame(audio_slice_pad, frame_length=self.audio_win, hop_length=self.audio_hop)                   

                    W = np.zeros_like(f)
                    for j in range(W.shape[1]):
                        W[:,j] = window
                    f = f*W
                    X.append(f.T)                 
                else:
                    X.append(audio_slice)  
                   

                #### Normalize by slicess
                stft = np.abs(librosa.core.stft(audio_slice_pad, n_fft=self.n_fft, hop_length=self.audio_hop,
                                                win_length=self.audio_win, center=False))**2
                stft = stft/(self.n_fft/2+1)
                S.append(stft)

                melspec = self.mel_basis.dot(stft)
                #melspec = melspec*self.alpha
                melspec = librosa.core.power_to_db(melspec)
                
                mel.append(melspec.T)

                # Get y
                j = int(i/float(self.sequence_hop_samples))

                if self.dataset == 'MAVD':
                    for index_list,l_list in enumerate(self.label_list):
                        y = event_rolls[index_list][j, :]
                        assert y.shape == (len(l_list),)
                        yt[index_list].append(y)
                        
                if self.dataset == 'URBAN-SED':
                    y = event_roll[j, :]
                    assert y.shape == (len(self.label_list),)
                    yt.append(y)
                    
                # Get id
                id = [ID, i]
                id_t.append(id)

        X = np.asarray(X)
        
        if self.dataset == 'MAVD':
            for index_list,l_list in enumerate(self.label_list):
                yt[index_list] = np.asarray(yt[index_list])
         
        if self.dataset == 'URBAN-SED':
            yt = np.asarray(yt)
         
        mel = np.asarray(mel)
        S = np.asarray(S)
        S = np.transpose(S,(0,2,1))

        X = np.expand_dims(X, -1)

        # Normalize
        if self.train:
            self.norm_scaler = StandardScaler()
            print(np.reshape(mel,(-1,self.mel_bands)).shape)
            self.norm_scaler.fit(np.reshape(mel,(-1,self.mel_bands)))
            assert len(self.norm_scaler.mean_) == self.mel_bands
            
            if self.normalize == 'standard':
                print('Normalize Standard')
                self.scaler = self.norm_scaler
                mel_dims = mel.shape
                mel_temp = mel.reshape(-1,self.mel_bands)
                mel_temp = self.scaler.transform(mel_temp)
                mel = mel_temp.reshape(mel_dims)
                
            if self.normalize == 'minmax':
                print('Normalize MinMax')
                min_v = np.amin(mel_all)#,axis=(0,2))
                max_v = np.amax(mel_all)#,axis=(0,2))
                self.scaler = [min_v,max_v]
                mel = 2*((mel-self.scaler[0])/(self.scaler[1]-self.scaler[0])-0.5)
        else:
            if self.normalize == 'standard': 
                mel = self.scaler.transform(mel) 
            if self.normalize == 'minmax':
                mel = 2*((mel-self.scaler[0])/(self.scaler[1]-self.scaler[0])-0.5)
     
        return X,S,mel,yt
    def return_all(self):

        """ This function generates data with all the files in self.list_IDs

        Return
        ----------
        X : array
            Audio signals (only for end-to-end networks)
            
        S : array
            Spectrograms
            
        mel : array
            Mel-spectrograms
           
        yt : array
            Annotationes as categorical matrix

        """        
        
        X,S,mel,y = self.__data_generation(self.list_IDs)

        return X,S,mel,y

    def return_random(self):

        """ This function generates data for a random file in self.list_IDs

        Return
        ----------
        X : array
            Audio signals (only for end-to-end networks)
            
        S : array
            Spectrograms
            
        mel : array
            Mel-spectrograms
           
        yt : array
            Annotationes as categorical matrix

        """       
        
        j = random.randint(0,len(self.list_IDs)-1)
        
        X,S,mel,y = self.__data_generation([self.list_IDs[j]])

        return X,S,mel,y

    def get_scaler(self):
        return self.scaler

    def get_standard_scaler(self):
        return self.norm_scaler
