# -*- coding: utf-8 -*-
"""
Implementation of epsilon-SVR
"""
import numpy as np
#import matplotlib.pyplot as plt
from openopt import QP
import math

class SVR():
	def __init__(self, epsilon=0.1, C=100):
		self.eps = epsilon
		self.C = C
	
	def _kernel(self, x, y):
		res = math.exp(-1 * abs(x - y) ** 2)
		return res
	
	def _product(self, a, X, x):
		res=0.0
		for i in range(len(a)):
			res = res + a[i] * self._kernel(X[i], x)
		return res
	
	def fit(self, X, Y):
		
		xdim = X.shape[0]
		ydim = Y.shape[0]
		
		if xdim != ydim:
			raise Exception('Input dimension does not match!')
		
		kernel=[[0.0 for i in range(2*xdim)] for j in range(2*xdim)]
		
		for i in range(xdim):
			for j in range(xdim):
				kernel[i][j] = self._kernel(X[i], X[j])
				kernel[i + xdim][j + xdim] = self._kernel(X[i], X[j])
		
		#----------------------------------------------------------------------------------------------------------------
		#negating the values for a_n'
		for i in range(xdim):
			for j in range(xdim):
				kernel[i + xdim][j] = (-1.0) * self._kernel(X[i], X[j])
				kernel[i][j + xdim] = (-1.0) * self._kernel(X[i], X[j])
		
		#--------------------------------------------------------------------------------------------------------------
		#coeff of 2nd term to minimize
		f=[0.0 for i in range(2 * xdim)]
		for i in range(xdim):
			f[i] = -float(Y[i]) + self.eps
		for i in range(xdim, 2 * xdim):
			f[i] = float(Y[i - xdim])+ self.eps
		
		#-----------------------------------------------------------------------------------------------------
		#constraints
		lower_limit = [0.0 for i in range(2 * xdim)]
		upper_limit = [float(self.C) for i in range(2 * xdim)]
		Aeq = [1.0 for i in range(2 * xdim)]
		for i in range(xdim, 2 * xdim):
			Aeq[i]=-1.0
		beq=0.0
			
		#----------------------------------------------------------------------------------------------------
		
		#coeff for 3rd constraint
		#kernel=H
		eq = QP(np.asmatrix(kernel),np.asmatrix(f),
		  lb=np.asmatrix(lower_limit),
		  ub=np.asmatrix(upper_limit),
		  Aeq=Aeq,beq=beq)
		p = eq._solve('cvxopt_qp', iprint = 0)
		f_opt, x = p.ff, p.xf
		
		#---------------------------------------------------------------------------------------
		self.support_vectors=[]
		self.support_vectors_Y=[]

		self.coeff=[]
		#support vectors: points such that an-an' ! = 0
		for i in range(xdim):
			if not((x[i]-x[xdim+i])==0):
				self.support_vectors.append( X[i] )
				self.support_vectors_Y.append(Y[i])
				self.coeff.append( x[i]-x[xdim+i] )
		
		
		low=min(abs(x))
		for i in range(xdim):
			if not(abs(x[i]-x[xdim+i]) < low + 0.005):
				self.support_vector.append( X[i] )
				self.support_vector_Y.append(Y[i])
				
		
		bias=0.0
		for i in range(len(X)):
			bias=bias+float(Y[i] - self.eps - self._product(coeff, self.support_vectors, X[i]))
		#generally bias is average as written in the book
		self.bias=bias/len(X)
		
	def predict(self, X):
		
		Y=[]
		eps_down = []
		eps_up = []
		
		for sample in X:
			Y.append(self._product(self.coeff, self.support_vectors, sample) + self.bias)
			
			eps_down.append(self._product(self.coeff,self.support_vectors,sample)+self.bias-self.eps)
			
			eps_up.append(self._product(self.coeff,self.support_vectors,sample)+self.bias+self.eps)
		
		return np.array(Y), np.array(eps_down), np.array(eps_up)

	



