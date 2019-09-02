from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
eps = 1e-6


# Metrics
def F1(y_true, y_pred):
    """
    Function that calculates the F1 metric for SED [1]
    
    [1] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 
    “Metrics for polyphonic sound event detection”, 
    Applied Sciences, 6(6):162, 2016
    
    ----------
    y_true : array
        Ground-truth of the evaluation
    
    y_pred : array
        Prediction of the system to be evaluated (predict_proba)
        
    Returns
    -------
    Fmeasure : float
        F1 value

    Notes
    -----
    Code based on sed_eval implementation
    http://tut-arg.github.io/sed_eval/
    
    """
    
    y_pred = (y_pred>0.5).astype(int)
    Ntp = np.sum(y_pred + y_true > 1)
    Ntn = np.sum(y_pred + y_true > 0)
    Nfp = np.sum(y_pred - y_true > 0)
    Nfn = np.sum(y_true - y_pred > 0)
    Nref = np.sum(y_true)
    Nsys = np.sum(y_pred)
    
    P = Ntp / float(Nsys + eps)
    R = Ntp / float(Nref + eps)

    Fmeasure = 2*P*R/(P + R + eps)
    return Fmeasure
    

def ER(y_true, y_pred):
    """
    Function that calculates the error rate (ER) metric for SED [1]
    
    [1] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 
    “Metrics for polyphonic sound event detection”, 
    Applied Sciences, 6(6):162, 2016
    
    ----------
    y_true : array
        Ground-truth of the evaluation
    
    y_pred : array
        Prediction of the system to be evaluated (predict_proba)
        
    Returns
    -------
    Fmeasure : float
        ER value

    Notes
    -----
    Code based on sed_eval implementation
    http://tut-arg.github.io/sed_eval/
    
    """    
    
    y_pred = (y_pred>0.5).astype(int)
    Ntp = np.sum(y_pred + y_true > 1)
    Nref = np.sum(y_true)
    Nsys = np.sum(y_pred)    
    
    S = min(Nref, Nsys) - Ntp
    D = max(0.0, Nref - Nsys)
    I = max(0.0, Nsys - Nref)    

    ER = (S+D+I)/float(Nref + eps)
    
    return(ER)   

class MetricsCallback(Callback):
    """Keras callback to calculate F1 and ER after each epoch and save 
    file with the weights if the evaluation improves
    """
    
    def __init__(self, x_val, y_val, f1s_current, f1s_best, file_weights):
        """ Initialize the keras callback
        Parameters
        ----------
        x_val : array
            Validation data for model evaluation
            
        y_val : array
            Ground-truth of th validation set
            
        f1s_current : float
            Last F1 value 
            
        f1s_current : float
            Best F1 value
            
        file_weights : string
            Path to the file with the weights
        """
        
        self.x_val = x_val
        self.y_val = y_val
        self.f1s_current = f1s_current
        self.f1s_best = f1s_best
        self.file_weights = file_weights
        self.epochs_since_improvement = 0
        self.epoch_best = 0

    def on_epoch_end(self, epoch, logs={}):
        """ This function is run when each epoch ends.
        The metrics F1 and ER are calcualted, printed and saved to the log file.
        Parameters
        ----------
        epoch : int
            number of epoch (from Callback class)
            
        logs : dict
            log data (from Callback class)

        """
        
        y_pred = self.model.predict(self.x_val)
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
            y_val = self.y_val[0]
        else:
            y_val = self.y_val       
        F = F1(y_val,y_pred)
        E = ER(y_val,y_pred)
        logs['F1'] = F
        logs['ER'] = E

        self.f1s_current = F

        if self.f1s_current > self.f1s_best:
            self.f1s_best = self.f1s_current
            self.model.save_weights(self.file_weights)
            print('F1 = {:.4f}, ER = {:.4f} -  Best val F1s: {:.4f} (IMPROVEMENT, saving)\n'.format(F, E, self.f1s_best))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('F1 = {:.4f}, ER = {:.4f} - Best val F1s: {:.4f} ({:d})\n'.format(F, E, self.f1s_best, self.epoch_best))
            self.epochs_since_improvement += 1
            

class MetricsCallback_levels(Callback):
    """Same that MetricsCallback but for the MAVD dataset (multiple outputs)"""
    def __init__(self, x_val, y_val, f1s_current, f1s_best, file_weights, pond=False, maxF_ER=False):
        self.x_val = x_val
        self.y_val = y_val
        self.f1s_current = f1s_current
        self.f1s_best = f1s_best
        self.er_current = 200
        self.er_best = 200
        self.file_weights = file_weights
        self.epochs_since_improvement = 0
        self.epoch_best = 0
        self.pond = pond
        self.maxF_ER = maxF_ER
        if pond == True:
            w1 = np.sum(y_val[0],axis=0)
            w1[w1==0] = np.amin(w1[np.nonzero(w1)])
            w1 = 1/w1/np.sum(1/w1)
            w2 = np.sum(y_val[1],axis=0)
            w2[w2==0] = np.amin(w2[np.nonzero(w2)])
            w2 = 1/w2/np.sum(1/w2)
            w3 = np.sum(y_val[2],axis=0)
            w3[w3==0] = np.amin(w3[np.nonzero(w3)])
            w3 = 1/w3/np.sum(1/w3)
            self.w = [w1,w2,w3]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        Fs = np.zeros(len(y_pred))
        ERs = np.zeros(len(y_pred))
        for j in range(len(y_pred)):
            if self.pond is False:
                Fs[j] = F1(self.y_val[j],y_pred[j])
                ERs[j] = ER(self.y_val[j],y_pred[j])
            else:
                ERc  = np.zeros(y_pred[j].shape[1])
                F1c  = np.zeros(y_pred[j].shape[1])
                for c in range(y_pred[j].shape[1]):
                    ERc[c] = ER(self.y_val[j][:,c],y_pred[j][:,c])
                    F1c[c] = F1(self.y_val[j][:,c],y_pred[j][:,c])
               # print(F1c)
                Fs[j] = np.sum(F1c*self.w[j])
                ERs[j] = np.sum(ERc*self.w[j])
        logs['F1'] = Fs
        logs['ER'] = ERs
        #print(Fs)
        if self.maxF_ER:
            self.f1s_current = np.mean(Fs)/np.mean(ERs)
        else:
            self.f1s_current = np.mean(Fs)
        self.er_current = np.mean(ERs)        
        #self.model.save_weights('weights_final.hdf5') # Graba SIEMPRE!!

        if  self.f1s_current > self.f1s_best: #self.er_best > self.er_current: 
            self.f1s_best = self.f1s_current
            #self.er_best = self.er_current
            self.model.save_weights(self.file_weights)
            print('F1 = {:.4f}, ER = {:.4f}, F1 = {:.4f}, ER = {:.4f}, F1 = {:.4f}, ER = {:.4f} -  Best val F1s: {:.4f} (IMPROVEMENT, saving)\n'.format(Fs[0], ERs[0],Fs[1], ERs[1],Fs[2], ERs[2], self.f1s_best))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('F1 = {:.4f}, ER = {:.4f}, F1 = {:.4f}, ER = {:.4f}, F1 = {:.4f}, ER = {:.4f}  - Best val F1s: {:.4f} ({:d})\n'.format(Fs[0], ERs[0],Fs[1], ERs[1],Fs[2], ERs[2], self.f1s_best, self.epoch_best))
            self.epochs_since_improvement += 1