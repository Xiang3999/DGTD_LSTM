#!/usr/bin/env python
# -*- coding: UTF-8 -*-


class error():
    def __init__(self, Nttr=263):
        self.mortimereErrorEz = np.zeros(Nttr)
        self.mortimereErrorHy = np.zeros(Nttr)  
        self.portimereErrorEz = np.zeros(Nttr)  
        self.portimereErrorHy = np.zeros(Nttr)  
error = error()

def getErr(Nttr, Ndof):
    '''
    Input: the number of time instances Nttr, 
           the degree of freedom Ndof.
    Output: the relative error between the DGTD and POD-DL-ROM solution
    '''
    mortimeErrorL2 = zeros(1,Nttr)
    protimeErrorL2 = zeros(1,Nttr)
    zeronDGTDTime = np.zeros(Ndof,2) # to compute the relative error
    for i in range(Nttr):
        MORTime = np.hstack((MORtime.Hye[:, i],MORtime.Eze[:, i]))
        DGTDTime = np.hstack((DGTDtime.Hye[:, i],DGTDtime.Eze[:, i]))  # snapshot
        proMORTime = np.hstack((timeparameterPOD.Basis.Hy*(np.transpose(timeparameterPOD.Basis.Hy)*DGTDtime.Hye[:,i]),\
                     timeparameterPOD.Basis.Ez*(np.transpose(timeparameterPOD.Basis.Ez)*DGTDtime.Eze[:,i])))
        morerrHy, morerrEz = computeErr(MORTime, DGTDTime)
        proerrHy, proerrEz = computeErr(proMORTime, DGTDTime)
        reproerrE, reproerrH = computeErr(zeronDGTDTime,DGTDTime)
        error.mortimereErrorEz[i] = morerrEz/reproerrEz;
        error.mortimereErrorHy[i] = morerrHy/reproerrHy;
        error.protimereErrorEz[i] = proerrEz/reproerrEz;
        error.protimereErrorHy[i] = proerrHy/reproerrHy;
        
    return error