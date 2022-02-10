# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 01:20:07 2021

@author: tom9m
"""

# -*- coding: utf-8 -*-

import math
import torch
import tomgpytorch
import sys
from matplotlib import pyplot as plt


import numpy as np  
from scipy.stats import zscore
data=np.loadtxt('../annotation/aist_data_fps3_gplvm.txt').astype(np.float32)
frame= np.loadtxt('../annotation/aist_framedata_fps3.txt')
data= zscore(data)
train=torch.from_numpy(data) 
f_count=0
name='aist4'
K=4
for n in range(3):
    m_list=[]
    v_list=[]
    w_list=[]
    f_count=0
    
    #動画数分回す
    for i in range(len(frame)):
        train_x=torch.linspace(0,1,int(frame[i]))
        train_y=train[f_count:f_count+int(frame[i]),n]   
        f_count +=int(frame[i])
        class SpectralMixtureGPModel(tomgpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = tomgpytorch.means.ConstantMean()
                self.covar_module = tomgpytorch.kernels.SpectralMixtureKernel(num_mixtures=K)
                self.covar_module.initialize_from_data(train_x, train_y)
        
            def forward(self,x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return tomgpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
            def m(self):
                return self.covar_module.return_m()
            def v(self):
                return self.covar_module.return_v()
            def w(self):
                return self.covar_module.return_w()
            
        likelihood = tomgpytorch.likelihoods.GaussianLikelihood()
        model = SpectralMixtureGPModel(train_x, train_y, likelihood)
        # this is for running the notebook in our testing framework
        import os
        smoke_test = ('CI' in os.environ)
        training_iter = 2 if smoke_test else 100
        
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = tomgpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for ii in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            sys.stdout.write("\r%i:%d / %d" % (i+1,ii + 1, training_iter))
            sys.stdout.flush()    
            optimizer.step()
        
        
        # Test points every 0.1 between 0 and 5
        test_x = torch.linspace(0, 1, 51)
        
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        m=model.m()
        v=model.v()
        w=model.w()       
        a=m.to('cpu').detach().numpy().copy().reshape(1,K)
        b=v.to('cpu').detach().numpy().copy().reshape(1,K)
        c=w.to('cpu').detach().numpy().copy().reshape(1,K)
        m_list.append(a[0])
        v_list.append(b[0])
        w_list.append(c[0])
        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        
    m_list=np.array(m_list)
    v_list=np.array(v_list)
    w_list=np.array(w_list)
    #print(m_list.shape)
    np.savetxt('../input/smkdata/{}_{}d.txt'.format(name,n+1),m_list)
    np.savetxt('../input/smkdata/{}_v{}d.txt'.format(name,n+1),v_list)
    np.savetxt('../input/smkdata/{}_w{}d.txt'.format(name,n+1),w_list)

"""
        if(i==2):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Make predictions
                observed_pred = likelihood(model(test_x))
            
                # Initialize plot
                f, ax = plt.subplots(1, 1, figsize=(4, 3))
            
                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                # Plot training data as black stars
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                # Plot predictive means as blue line
                ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.set_ylim([-2, 2])
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
"""