import numpy as np
from numba import njit 

@njit('float64(float64[:])',cache=True)
def _logsumexp(a):
    a_max = np.max(a)

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64(float64[:],float64[:])',cache=True)
def _logsumexpb(a,b):

    a_max = np.max(a)

    tmp = b * np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64(float64)',cache=True)
def _log_phi(z):
	logphi = -0.5 * np.log(2.0* np.pi) - 0.5 * z * z
	return logphi


@njit('float64[:,:](float64[:,:],float64[:,:])',cache=True)
def _log_prob_mat_mul(A,B):
    # multiplication of probability matrices in log space
    C = np.zeros((A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = _logsumexp( A[i,:] + B[:,j])
            if np.isnan(C[i,j]):
                C[i,j] = np.NINF
        C[i,:] -= _logsumexp(C[i,:])
    return C

@njit('float64[:,:](float64[:,:],int64)',cache=True)
def _log_matrix_power(X,n):
    ## find log of exp(X)^n (pointwise exp()!) 

    # use 18 because you are fucked if you want trans
    # for dt > 2^18...
    #print('Calculating matrix powers...')

    maxlog2dt = 18
    assert(np.log(n)/np.log(2) < maxlog2dt)
    assert(X.shape[0] == X.shape[1])
    b = 1
    k = 0
    matrices = np.zeros((X.shape[0],X.shape[1],maxlog2dt))
    matrices[:,:,0] = X
    
    while b < n:
        #print(b,k)
        k += 1
        b += 2**k
        # square the last matrix
        matrices[:,:,k] = _log_prob_mat_mul(matrices[:,:,k-1],
                                           matrices[:,:,k-1])
    leftover = n
    Y = np.NINF * np.ones((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        Y[i,i] = 0
        
    while leftover > 0:
        #print(n-leftover,k)
        if 2**k <= leftover:
            Y = _log_prob_mat_mul(Y,matrices[:,:,k])
            leftover -= 2**k
        k -= 1
        
    return Y

@njit('float64[:](int64,float64,float64,float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _log_trans_prob(i,N,s,FREQS,z_bins,z_logcdf,z_logsf,dt):
	# 1-generation transition prob based on Normal distn
	
	p = FREQS[i]
	lf = len(FREQS)
	logP = np.NINF * np.ones(lf)

	if p <= 0.0:
		logP[0] = 0
	elif p >= 1.0:
		logP[lf-1] = 0
		return logP
	else:

		if s != 0:
			mu = p - s*p*(1.0-p)*dt

		else:
			mu = p 

		sigma = np.sqrt(p*(1.0-p)/(4.0*N)*dt)

                      
		pi0 = np.interp(np.array([(FREQS[0]-mu)/sigma]),z_bins,z_logcdf)[0]
		pi1 = np.interp(np.array([(FREQS[lf-1]-mu)/sigma]),z_bins,z_logsf)[0]

		x = np.array([0.0,pi0,pi1])
		b = np.array([1.0,-1.0,-1.0])
		middleNorm = _logsumexpb(x,b)

		middleP = np.zeros(lf-2)
		for j in range(1,lf-1):
			if j == 1:
				mlo = FREQS[0]
			else:
				mlo = np.mean(np.array([FREQS[j],FREQS[j-1]]))
			if j == lf-2:
				mhi = FREQS[j+1]
			else:
				mhi = np.mean(np.array([FREQS[j],FREQS[j+1]]))
    
			l1 = np.interp(np.array([(mlo-mu)/sigma]),z_bins,z_logcdf)[0]
			l2 = np.interp(np.array([(mhi-mu)/sigma]),z_bins,z_logcdf)[0]
			middleP[j-1] = _logsumexpb(np.array([l1,l2]),np.array([-1.0,1.0]))                    


		logP[0] = pi0
		logP[1:lf-1] = middleP
		logP[lf-1] = pi1

	return logP

@njit('float64[:,:](float64,float64,float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _nstep_log_trans_prob(N,s,FREQS,z_bins,z_logcdf,z_logsf,dt):
	lf = len(FREQS)
	p1 = np.zeros((lf,lf))

	# load rows into p1
	for i in range(lf):
		row = _log_trans_prob(i,N,s,FREQS,z_bins,z_logcdf,z_logsf,1)
		p1[i,:] = row

	# exponentiate matrix
	pn = _log_matrix_power(p1,int(dt))
	return pn

@njit('float64(float64[:],float64)')
def _genotype_likelihood_emission(ancGLs,p):
	logGenoFreqs = np.array([2*np.log(1-p),np.log(2) + np.log(p) + np.log(1-p),2*np.log(p)])
	emission = _logsumexp(logGenoFreqs + ancGLs)
	if np.isnan(emission):
		emission = -np.inf
	return emission

@njit('float64(float64[:],int64,float64[:],float64,float64,float64,int64)',cache=True)
def _log_coal_density(times,n,epoch,xi,Ni,N0,anc=0):
    if n == 1:
        # this flag indicates to ignore coalescence
        return 0.0
    
    logp = 0
    prevt = epoch[0]
    if anc == 1:
        xi = 1.0-xi
    k=n
    for i,t in enumerate(times):
        k = n-i
        kchoose2 = k*(k-1)/4
        dLambda = 1/(xi*Ni)*(t-prevt)
        logpk = - np.log(xi) - kchoose2 * dLambda
        logp += logpk
        
        prevt = t
        k -= 1
    kchoose2 = k*(k-1)/4
    logPk = - kchoose2 * 1/(xi*Ni)*(epoch[1]-prevt)

    logp += logPk
    return logp

@njit('float64[:,:](float64[:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:],int64)',cache=True)
def forward_algorithm(sel,times,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,noCoals=1):

    '''
    Moves forward in time from past to present
    '''
    
    lf = len(freqs)
    
    alpha = np.zeros(lf)
    alpha -= _logsumexp(alpha)
    
    T = len(epochs)-1
    alphaMat = np.zeros((T+1,lf))
    alphaMat[-1,:] = alpha

    prevNt = -1
    prevst = -1
    prevdt = -1
    prevNumDerCoals = -1
    prevNumAncCoals = -1
    
    cumGens = epochs[-1]
    
    nDer = np.sum(times[0,:]>=0)+1
    nDerRemaining = nDer - np.sum(np.logical_and(times[0,:]>=0, times[0,:]<=epochs[-1]))
    nAnc = np.sum(times[1,:]>=0)+1
    nAncRemaining = nAnc - np.sum(np.logical_and(times[1,:]>=0, times[1,:]<=epochs[-1]))
    coalEmissions = np.zeros(lf)
    N0 = N[0]
    for tb in range(T-1,0,-1):
        dt = -epochs[tb]+epochs[tb+1]
        epoch = np.array([cumGens - dt,cumGens])
        Nt = N[tb]
        
        st = sel[tb]
        prevAlpha = np.copy(alpha)
        
        if prevNt != Nt or prevst != st or prevdt != dt:
            #print(Nt,st,dt)
            currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
        
        #grab ancient GL rows
        ancientGLrows = ancientGLs[np.logical_and(ancientGLs[:,0] <= cumGens, ancientGLs[:,0] > cumGens - dt)]
        
        # calculate ancient GL emission probs
        glEmissions = np.zeros(lf)
        
        for j in range(lf):
            for iac in range(ancientGLrows.shape[0]):
                glEmissions[j] += _genotype_likelihood_emission(ancientGLrows[iac,1:],freqs[j])
                
        # calculate coal emission probs
        
        if noCoals:
            coalEmissions = np.zeros(lf)
        else:
            derCoals = np.copy(times[0,:])
            derCoals = derCoals[derCoals <= cumGens]
            derCoals = derCoals[derCoals > cumGens-dt]
            numDerCoals = len(derCoals)
            ancCoals = np.copy(times[1,:])
            ancCoals = ancCoals[ancCoals <= cumGens]
            ancCoals = ancCoals[ancCoals > cumGens-dt]
            numAncCoals = len(ancCoals)
            nDerRemaining += len(derCoals)
            nAncRemaining += len(ancCoals)
            #print(epoch,derCoals,nDerRemaining,nAncRemaining)
            #if prevNt != Nt or prevst != st or prevdt != dt or numDerCoals != 0 or prevNumAncCoals != 0 or numAncCoals != 0 or prevNumAncCoals != 0:
                #print(cumGens)
            for j in range(lf):
                    coalEmissions[j] = _log_coal_density(derCoals,nDerRemaining,epoch,freqs[j],Nt,N0,anc=0)
                    coalEmissions[j] += _log_coal_density(ancCoals,nAncRemaining,epoch,freqs[j],Nt,N0,anc=1)

        
        #print(tb,ancientGLrows)
        for i in range(lf):
            alpha[i] = _logsumexp(prevAlpha + currTrans[i,:] + glEmissions + coalEmissions) 
            if np.isnan(alpha[i]):
                alpha[i] = -np.inf
        
        prevNt = Nt
        prevdt = dt
        prevst = st
        prevNumAncCoals = numAncCoals
        prevNumDerCoals = numDerCoals
        cumGens -= dt
        alphaMat[tb,:] = alpha
    return alphaMat
    
@njit('float64[:,:](float64[:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:],int64,float64)',cache=True)
def backward_algorithm(sel,times,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,noCoals=1,currFreq=-1):

    '''
    Moves backward in time from present to past
    '''
    
    lf = len(freqs)
    alpha = np.zeros(lf)
    
    if currFreq != -1:
        nsamp = 1000
        for i in range(lf):
            k = int(currFreq*nsamp)
            alpha[i] = -np.sum(np.log(np.arange(2,k+1)))
            alpha[i] += np.sum(np.log(np.arange(2,nsamp-k+1)))
            alpha[i] += k*np.log(freqs[i]) + (nsamp-k)*np.log(1-freqs[i])
            
    T = len(epochs)-1
    alphaMat = np.zeros((T+1,lf))
    alphaMat[0,:] = alpha
    
    prevNt = -1
    prevst = -1
    prevdt = -1
    prevNumDerCoals = -1
    prevNumAncCoals = -1
    
    cumGens = 0
    
    nDer = np.sum(times[0,:]>=0)+1
    nDerRemaining = nDer
    nAnc = np.sum(times[1,:]>=0)+1
    nAncRemaining = nAnc
    N0 = N[0]
    coalEmissions = np.zeros(lf)
    for tb in range(0,T):
        dt = epochs[tb+1]-epochs[tb]
        Nt = N[tb]
        epoch = np.array([cumGens,cumGens+dt])
        st = sel[tb]
        prevAlpha = np.copy(alpha)
        
        if prevNt != Nt or prevst != st or prevdt != dt:
            #print(Nt,st,dt)
            currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
        
        #grab ancient GL rows
        ancientGLrows = ancientGLs[ancientGLs[:,0] > cumGens]
        ancientGLrows = ancientGLrows[ancientGLrows[:,0] <= cumGens + dt]

        glEmissions = np.zeros(lf)
        for j in range(lf):
            for iac in range(ancientGLrows.shape[0]):
                glEmissions[j] += _genotype_likelihood_emission(ancientGLrows[iac,1:],freqs[j])
        
        #grab coal times during epoch
        # calculate coal emission probs
        if noCoals:
            coalEmissions = np.zeros(lf)
            numAncCoals = -1
            numDerCoals = -1
        else:            
            derCoals = np.copy(times[0,:])
            derCoals = derCoals[derCoals > cumGens]
            derCoals = derCoals[derCoals <= cumGens+dt]
            numDerCoals = len(derCoals)
            ancCoals = np.copy(times[1,:])
            ancCoals = ancCoals[ancCoals > cumGens]
            ancCoals = ancCoals[ancCoals <= cumGens+dt]
            numAncCoals = len(ancCoals)
            #print(epoch,derCoals,nDerRemaining,nAncRemaining)
            #if prevNt != Nt or prevst != st or prevdt != dt or numDerCoals != 0 or prevNumAncCoals != 0 or numAncCoals != 0 or prevNumAncCoals != 0:
            for j in range(lf):
                    coalEmissions[j] = _log_coal_density(derCoals,nDerRemaining,epoch,freqs[j],Nt,N0,anc=0)
                    coalEmissions[j] += _log_coal_density(ancCoals,nAncRemaining,epoch,freqs[j],Nt,N0,anc=1)
            nDerRemaining -= len(derCoals)
            nAncRemaining -= len(ancCoals)

        
        
        #print(tb,ancientGLrows)
        for i in range(lf):
            alpha[i] = _logsumexp(prevAlpha + currTrans[:,i] ) + glEmissions[i] + coalEmissions[i]
            if np.isnan(alpha[i]):
                alpha[i] = -np.inf
        
        prevNt = Nt
        prevdt = dt
        prevst = st
        prevNumDerCoals = numDerCoals
        prevNumAncCoals = numAncCoals
        
        cumGens += dt
        alphaMat[tb,:] = alpha
    return alphaMat

@njit('float64(float64[:,:],float64[:],float64[:])',cache=True)
def proposal_density(times,epochs,N):
    '''
    Moves backward in time from present to past
    '''
    
    logl = 0.
    cumGens = 0
    T = len(epochs)-1
    combinedTimes = np.sort(np.concatenate((times[0,:],times[1,:])))
    n = np.sum(combinedTimes>=0)+2
    nRemaining = n
    N0 = N[0]
    for tb in range(0,T):
        dt = epochs[tb+1]-epochs[tb]
        Nt = N[tb]
        epoch = np.array([cumGens,cumGens+dt])
        
        #grab coal times during epoch
        # calculate coal emission probs

        Coals = np.copy(combinedTimes)
        Coals = Coals[Coals > cumGens]
        Coals = Coals[Coals <= cumGens+dt]
        numCoals = len(Coals)
      
        logl += _log_coal_density(Coals,nRemaining,epoch,1.0,Nt,N0,anc=0)
        nRemaining -= len(Coals)
        
        cumGens += dt
    return logl
