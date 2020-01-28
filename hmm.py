import numpy as np
from numba import njit #,jit,int32,float32,int64,float64,typeof

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

# cdef Phi(double z):
# 	cdef double a1 =  0.254829592
# 	cdef double a2 = -0.284496736
# 	cdef double a3 =  1.421413741
# 	cdef double a4 = -1.453152027
# 	cdef double a5 =  1.061405429
# 	cdef double p  =  0.3275911

# 	cdef int sign = 1
# 	if (z<0):
# 		sign = -1
# 	z = fabs(z)/sqrt(2.0)

# 	cdef double t = 1.0/(1.0 + p*z)
# 	cdef double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-z*z)
# 	return 0.5 * (1.0 + sign * y)

@njit('float64(float64,float64)',cache=True)
def _eta(alpha,beta):
	return alpha * beta / (alpha * (np.exp(beta) - 1) + beta * np.exp(beta))

@njit('float64(float64,float64,float64,float64)',cache=True)
def _griffiths_log_prob_coal_counts(a,b,t,N):
	t *= 1.0/(2.0*N)
	n = a
	alpha = 0.5*n*t
	beta = -0.5*t
	h = _eta(alpha,beta)
	mu = 2.0 * h * t**(-1.0)
	var = 2.0*h*t**(-1.0) * (h + beta)**2
	var *= (1.0 + h/(h+beta) - h/alpha - h/(alpha + beta) - 2.0*h)
	var *= beta**-2

	std = np.sqrt(var)
	lp = np.zeros(int(a))
	for bprime in range(1,int(a)+1):
		lp[bprime-1] = _log_phi((float(bprime)-mu)/std)

	return _log_phi((b-mu)/std) - _logsumexp(lp)

@njit('float64(float64,float64,float64,float64)',cache=True)
def _tavare_log_prob_coal_counts(a, b, t, n):
	#print('hi')
	lnC1 = 0.0
	lnC2=0.0
	C3=1.0

	for y in range(0,int(b)):
		lnC1 += np.log((b+y)*(a-y)/(a+y))

	s = -b*(b-1.0)*t/4.0/n

	for k in range(int(b)+1,int(a)+1):
		k1 = k - 1.0
		lnC2 += np.log((b+k1)*(a-k1)/(a+k1)/(k-b))
		C3 *= -1.0
		val = -k*k1*t/4.0/n + lnC2 + np.log((2.0*k-1.0)/(k1+b))
		if True:
			loga = s
			logc = val
			if (logc > loga):
				tmp = logc
				logc = loga
				loga = tmp
			s = loga + np.log(1.0 + C3*np.exp(logc - loga))
	for i in range(2,int(b)+1):
		s -= np.log(i)

	return s + lnC1

@njit('float64(float64[:],int64,float64,float64,float64,int64)',cache=True)
def _structured_coal(times,n,Nnow,x,t,anc=0):
	logCondProb = 0.0
	prevt = 0.0
	if anc:
		x = 1-x
	if n <= 1:
		return logCondProb
	else:
		if x == 0.0:
			return -np.inf
	for i,ti in enumerate(times):
		k = n-i
		k1 = k-1
		dt = ti-prevt
		logCondProb += np.log(k*k1/4.0/Nnow) - np.log(x) 
		logCondProb += -k*k1/4.0/Nnow * dt/x
		prevt = ti
	dt = t-prevt
	logCondProb += -k*k1/4.0/Nnow * dt/x
	return logCondProb

@njit('float64(int64,int64,int64,int64,int64,int64,float64,float64,float64,int64)',cache=True)
def _tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nnow,x,t,isTavare):

		logCondProb = 0.0
		if x == 0.0:
				if Cder1 > 1:
						return -np.inf
				if Cmix0 == 1:
						return 0.0
				b = Cmix1
				a = Cmix0

				if isTavare:
					logCondProb += _tavare_log_prob_coal_counts(float(a),float(b),t,Nnow)
				else:
					logCondProb += _griffiths_log_prob_coal_counts(float(a),float(b),t,Nnow)
		else:
				if Cmix1 == Canc1 and Cder0 > 0.0:
						return -np.inf

				for (N,a,b) in zip([Nnow*x,Nnow*(1-x)],[Cder0,Canc0],[Cder1,Canc1]):
						if a == 1 or a == 0:
								continue
						elif N == 0:
								return -np.inf
						if isTavare:
							logCondProb += _tavare_log_prob_coal_counts(float(a),float(b),t,N)
						else:
							logCondProb += _griffiths_log_prob_coal_counts(float(a),float(b),t,N)
		return logCondProb

@njit('float64[:,:](float64[:,:],float64[:,:])',cache=True)
def _log_prob_mat_mul(A,B):
    # multiplication of probability matrices in log space
    C = np.zeros((A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = _logsumexp( A[i,:] + B[:,j])
            if np.isnan(C[i,j]):
                C[i,j] = np.NINF
        ## special sauce...
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
		#plo = (FREQS[i]+FREQS[i-1])/2
		#phi = (FREQS[i]+FREQS[i+1])/2
		if s != 0:
			mu = p - 2*s*p*(1.0-p)/np.tanh(N*s*(1-p))*dt
			# mulo = plo - s*plo*(1.0-plo)/np.tanh(2*N*s*(1-plo))*dt
			# muhi = phi - s*phi*(1.0-phi)/np.tanh(2*N*s*(1-phi))*dt
		else:
			mu = p - p * 1/(4.0*N)*dt
			# mulo = plo - plo * 1/(2.0*N)*dt
			# muhi = phi - phi * 1/(2.0*N)*dt
		sigma = np.sqrt(p*(1.0-p)/(4.0*N)*dt)
		# sigmalo = np.sqrt(plo*(1.0-plo)/(2.0*N)*dt)
		# sigmahi = np.sqrt(phi*(1.0-phi)/(2.0*N)*dt)
                      
		pi0 = np.interp(np.array([(FREQS[0]-mu)/sigma]),z_bins,z_logcdf)[0]
		pi1 = np.interp(np.array([(FREQS[lf-1]-mu)/sigma]),z_bins,z_logsf)[0]

		x = np.array([0.0,pi0,pi1])
		b = np.array([1.0,-1.0,-1.0])
		middleNorm = _logsumexpb(x,b)
        
		# x = np.array([0.0,pi0lo,pi1lo,0.0,pi0hi,pi1hi])
		# b = np.array([0.5,-0.5,-0.5,0.5,-0.5,-0.5])
		# middleNorm = _logsumexpb(x,b)

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

@njit('float64[:,:](float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:],int64,float64[:,:])',cache=True)
def _forward_algorithm(selOverTime,dts,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,tCutoff,ancientGLs):

	'''
	Moves backward in time from present to past
	'''

	prevst = -999999
	prevNt = -1
	prevdt = -1
	lf = freqs.shape[0]
	alpha = np.zeros(lf)
	prevAlpha = np.zeros(lf)
	T = len(selOverTime)
	#if fb:
	alphaMat = np.zeros((lf,T))
	
	currTrans = np.zeros((lf,lf))
	cumGens = 0
	D = C[0,:]
	A = C[1,:]
	n = len(D[D>=0])
	m = len(A[A>=0])
	for t,st in enumerate(selOverTime):
		Nt = N[t]
		dt = dts[t]
		cumGens += dt
		isTavare = int(cumGens > coalModelChangepoint)
		if prevNt != Nt or prevst != st or prevdt != dt:
			# recalculate freq transition matrix
			#for i in range(lf):
			#	row = _log_trans_prob(i,Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
			#	currTrans[i,:] = row

			currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
		derTimesSlice = D[(D < cumGens) & (D >= cumGens-dt)]
		ancTimesSlice = A[(A < cumGens) & (A>= cumGens-dt)]
		derTimes = derTimesSlice-cumGens+dt
		ancTimes = ancTimesSlice-cumGens+dt

		if t == 0:
			alpha = -np.inf * np.ones(lf)
			alpha[dpfi] = _structured_coal(derTimes,n,Nt,freqs[dpfi],dt,anc=0)
			alpha[dpfi] += _structured_coal(ancTimes,m,Nt,freqs[dpfi],dt,anc=1)
			#print(alpha)
			alphaMat[:,0] = alpha
		else:
			#if prevNt != Nt or prevst != st or prevdt != dt:
			#	print('Marginalizing hidden states...')
			if True:
				coalVec = np.zeros(lf)
				for j in range(lf):
					coalVec[j] = _structured_coal(derTimes,n,Nt,freqs[j],dt,anc=0)
					coalVec[j] += _structured_coal(ancTimes,m,Nt,freqs[j],dt,anc=1)
				#if np.sum(coalVec > 0) > 0:
				#	print(t,dt,coalVec)

			#grab ancient GL rows
			ancientGLrows = ancientGLs[ancientGLs[:,0] >= cumGens-dt]
			ancientGLrows = ancientGLrows[ancientGLrows[:,0] < cumGens]
			
			for j in range(lf):
				glEmission = 0
				#for ancGLrow in ancientGLrows:
				for iac in range(ancientGLrows.shape[0]):
					glEmission += _genotype_likelihood_emission(ancientGLrows[iac,1:],freqs[j])
				alpha[j] = _logsumexp(prevAlpha+currTrans[:,j]) + coalVec[j] + glEmission
				if np.isnan(alpha[j]):
					alpha[j] = -np.inf
			alphaMat[:,t] = alpha

		prevAlpha = alpha
		prevst = st
		prevNt = Nt
		prevdt = dt
		n -= len(derTimes)
		m -= len(ancTimes) 
		if cumGens >= tCutoff:
			break
	return alphaMat

@njit('float64[:,:](float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:],float64[:],int64,float64[:,:])',cache=True)
def _backward_algorithm(selOverTime,dts,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,tDeep,ancientGLs):

	'''
	Moves forward in time from present to past
	'''

	prevst = -999999
	prevNt = -1
	prevdt = -1
	lf = freqs.shape[0]
	beta = np.zeros(lf)
	T = len(selOverTime)
	betaMat = np.zeros((lf,T))
	betaMat[:,tDeep-1] = prevBeta

	currTrans = np.zeros((lf,lf))
	cumGens = np.sum(dts[:tDeep])
	n = 1 
	m = 1 
	D = C[0,:]
	A = C[1,:]

	for tprime,st in enumerate(selOverTime[:tDeep][::-1]):

		tcoal = tDeep-tprime
		t = tcoal-1
		dt = dts[t]
		cumGens -= dt

		Nt = N[tcoal-1]
		isTavare = int(np.sum(dts[:t]) > coalModelChangepoint)
		if prevNt != Nt or prevst != st or prevdt != dt:
			currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)

		derTimesSlice = D[(D > cumGens) & (D <= cumGens+dt)]
		ancTimesSlice = A[(A > cumGens) & (A <= cumGens+dt)]
		derTimes = derTimesSlice-cumGens
		ancTimes = ancTimesSlice-cumGens
		n += len(derTimes)
		m += len(ancTimes)
		#print(cumGens,n,m,derTimes,ancTimes)
		#print(cumGens,Nt)	
		if True: 
		
			coalVec = np.zeros(lf)
			for j in range(lf):
				coalVec[j] = _structured_coal(derTimes,n,Nt,freqs[j],dt,anc=0)
				coalVec[j] += _structured_coal(ancTimes,m,Nt,freqs[j],dt,anc=1) 
			#print(coalVec)
		#grab ancient GL rows
		ancientGLrows = ancientGLs[ancientGLs[:,0] >= cumGens]
		ancientGLrows = ancientGLrows[ancientGLrows[:,0] < cumGens + dt]
			
		glEmissions = np.zeros(lf)
		for j in range(lf):
			for iac in range(ancientGLrows.shape[0]):
				glEmissions[j] += _genotype_likelihood_emission(ancientGLrows[iac,1:],freqs[j])


		for i in range(lf):
			beta[i] = _logsumexp(prevBeta + currTrans[i,:] + coalVec + glEmissions)
			if np.isnan(beta[i]):
				beta[i] = -np.inf

		betaMat[:,t] = beta
			
		prevBeta = beta
		prevst = st
		prevNt = Nt
		prevdt = dt
	return betaMat

@njit('float64[:](float64[::1],float64[::1],float64[:,::1],float64[::1],int64,float64[::1],int64,float64[::1],float64[::1],float64[::1],float64[:,::1],int64)',cache=True)
def forward_backward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,
									z_bins,z_logcdf,z_logsf,ancientGLs,tDeep=5000):
	print('Running forward-backward...')

	lf = len(freqs)
	T = len(selOverTime)
	#T = tDeep

	alphaMat = _forward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,10**7,ancientGLs)

	prevBeta = np.NINF * np.ones(lf)
	prevBeta[0] = 0
	#prevBeta = np.zeros(lf)
	betaMat = _backward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,T,ancientGLs)
	logPostMarg = np.zeros((lf,T))
	xPostMean = np.zeros(T)
	for t in range(T):
		logPostMarg[:,t] = alphaMat[:,t] + betaMat[:,t] - _logsumexp(alphaMat[:,t] + betaMat[:,t])
		
		xPostMean[t] = np.sum(np.exp(logPostMarg[:,t]) * freqs)
		#xPostMean[t] = freqs[logPostMarg[:,t].argmax()]
	return xPostMean

@njit('float64(float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:],float64[:],int64,float64[:,:])',cache=True)
def _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,
							z_bins,z_logcdf,z_logsf,prevBeta,tDeep,ancientGLs):
	betaMat = _backward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,tDeep,ancientGLs)
	return betaMat[dpfi,0]

#@njit('Tuple((float64[:],float64[:]))(	float64[:], \
				# 						float64[:], \
				# 						float64[:], \
				# 						float64[:,:],\
				# 						float64[:],\
				# 						int64,\
				# 						float64[:],\
				# 						float64[:],\
				# 						float64[:],\
				# 						float64[:],\
				# 						int64)',
				# cache=True)
def likelihood_grid_const(S,times,diffTimes,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf,coalModelChangepoint,ancientGLs,timeDeep=40000):

	print('Calculating likelihood surface and MLE...')
	T = len(times)-1
	lf = len(freqs)

	# neutral likelihood
	# calculate full betas (betMatNeu) for reuse in other models
	prevBeta = np.NINF * np.ones(lf)
	prevBeta[0] = 0
	#prevBeta = np.zeros(lf)
	betaMatNeu = _backward_algorithm(np.zeros(T),diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,T,ancientGLs)
	logL = np.zeros(len(S))
	logL[0] = betaMatNeu[dpfi,0]

	tDeep = int(np.digitize(timeDeep,times))
	prevBeta = betaMatNeu[:,tDeep]
	print('[',int(100/len(S)),'%] logL at s = ',0,':',logL[0])
	for (im1,s) in enumerate(S[1:]):
		i = im1 + 1
		selOverTime = np.ones(T)
		selOverTime *= s
		logL[i] = _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,tDeep,ancientGLs)
		print('[',int(100*(i+1)/len(S)),'%] logL at s = ',s,':',logL[i])
	## run maximum-likelihood
	i_ml = np.argmax(logL)
	mleSelOverTime = np.zeros(T)
	mleSelOverTime[:tDeep] = S[i_ml]
	print(S[i_ml])

	return (logL, mleSelOverTime)

#@njit('Tuple((float64[:,:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],float64[:],float64[:],float64[:],int64,int64)',cache=True)
def likelihood_grid_pulse(S,pulseTimes,times,diffTimes,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf,coalModelChangepoint,pulseLen,ancientGLs,timeDeep=5000):

	print('Calculating likelihood surface and MLE...')
	T = len(times)-1
	lf = len(freqs)

	# test for a pulse at each time specified in pulseTimes
	prevBeta = np.NINF * np.ones(lf)
	prevBeta[0] = 0
	betaMatNeu = _backward_algorithm(np.zeros(T),diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,T,ancientGLs)
	logLp = np.zeros((len(S), len(pulseTimes)))	
	logLp[0,:] = betaMatNeu[dpfi,0]

	#timeDeep = pulseTimes[-1] + pulseLen 
	tDeep = int(np.digitize(timeDeep,times))
	prevBeta = betaMatNeu[:,tDeep]
	ctimes = np.ascontiguousarray(times)
	
	for j,tb in enumerate(pulseTimes):
		ipulse0 = int(np.digitize(np.array([tb]),ctimes)[0])
		ipulse1 = int(np.digitize(np.array([tb+pulseLen]),ctimes)[0])
		for im1,s in enumerate(S[1:]):
				i = im1+1
				selOverTime = np.zeros(T)
				selOverTime[ipulse0-1:ipulse1-1] = s
				logLp[i,j] = _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf,prevBeta,tDeep,ancientGLs)
				#print('[%d%%] logL at s = %.2e, t = %d-%d :\t%.2f'%(int(100*(i*len(pulseTimes) + j)/(len(S)*len(pulseTimes))),s,tb,tb+pulseLen,logL[i,j]))
				print('[',int(100*(i*len(pulseTimes) + j)/(len(S)*len(pulseTimes))),'%] logL at s = ',s,' , t = ',tb,':',logLp[i,j])
	## run maximum-likelihood
	iam = logLp.argmax()
	i_ml0 = iam // logLp.shape[1]
	i_ml1 = iam % logLp.shape[1]
	pt_ml = pulseTimes[i_ml1]
	ipulse0_ml = int(np.digitize(np.array([pt_ml]),ctimes)[0])
	ipulse1_ml = int(np.digitize(np.array([pt_ml+pulseLen]),ctimes)[0])
	mleSelOverTime = np.zeros(T)
	mleSelOverTime[ipulse0_ml-1:ipulse1_ml-1] = S[i_ml0]
	print(S[i_ml0],times[ipulse0_ml-1],times[ipulse1_ml-1])
	return (logLp, mleSelOverTime)

