import numpy as np
from hmm_utils import forward_algorithm
from hmm_utils import backward_algorithm
from hmm_utils import proposal_density
from scipy.special import logsumexp
import scipy.stats as stats
from scipy.optimize import minimize
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--times',type=str,
		help='Should refer to files <times>.{{der,anc}}.npy (exclude prefix .{{der,anc}}.npy)',
		default=None)
	parser.add_argument('--popFreq',type=float,default=None)
	parser.add_argument('-q','--quiet',action='store_true')
	parser.add_argument('-o','--output',dest='outFile',type=str,default=None)

	parser.add_argument('--ancientSamps',type=str,default=None)
	parser.add_argument('--out',type=str,default=None)
	#advanced options
	parser.add_argument('-N','--N',type=float,default=10**4)
	parser.add_argument('-coal','--coal',type=str,default=None,help='path to Relate .coal file. Negates --N option.')

	#parser.add_argument('--optimize',action='store_true')

	parser.add_argument('--stepsize',type=float,default=1e-5)
	parser.add_argument('--thresh',type=float,default=1e-4)	
	parser.add_argument('-thin','--thin',type=int,default=1)
	parser.add_argument('-burnin','--burnin',type=int,default=0)
	parser.add_argument('--tCutoff',type=float,default=1000)
	parser.add_argument('--timeBins',type=str,default=None)

	return parser.parse_args()


def load_normal_tables():
    # read in global Phi(z) lookups
    z_bins = np.genfromtxt('z_bins.txt')
    z_logcdf = np.genfromtxt('z_logcdf.txt')
    z_logsf = np.genfromtxt('z_logsf.txt')
    return z_bins,z_logcdf,z_logsf

def load_times(args):
	locusDerTimes = np.load('%s.der.npy'%(args.times))[:,:]
	locusAncTimes = np.load('%s.anc.npy'%(args.times))[:,:]

	if locusDerTimes.ndim == 0 or locusAncTimes.ndim == 0:
		raise ValueError
	if np.prod(locusDerTimes.shape) == 0 or np.prod(locusAncTimes.shape) == 0:
		raise ValueError	
	elif locusAncTimes.ndim == 1 and locusDerTimes.ndim == 1:
		M = 1
		locusDerTimes = np.transpose(np.array([locusDerTimes]))
		locusAncTimes = np.transpose(np.array([locusAncTimes]))
	elif locusAncTimes.ndim == 2 and locusDerTimes.ndim == 1:
		locusDerTimes = np.array([locusDerTimes])[:,args.burnin::args.thin]
		locusAncTimes = np.transpose(locusAncTimes)[:,args.burnin::args.thin]
		M = locusDerTimes.shape[1]	
	elif locusAncTimes.ndim == 1 and locusDerTimes.ndim == 2:
		locusAncTimes = np.array([locusAncTimes])[:,args.burnin::args.thin]
		locusDerTimes = np.transpose(locusDerTimes)[:,args.burnin::args.thin]
		M = locusDerTimes.shape[1]
	else:
		locusDerTimes = np.transpose(locusDerTimes)[:,args.burnin::args.thin]
		locusAncTimes = np.transpose(locusAncTimes)[:,args.burnin::args.thin]
		M = locusDerTimes.shape[1]
	n = locusDerTimes.shape[0] + 1
	m = locusAncTimes.shape[0] + 1
	ntot = n + m
	row0 = -1.0 * np.ones((ntot,M))
	row0[:locusDerTimes.shape[0],:] = locusDerTimes

	row1 = -1.0 * np.ones((ntot,M))
	row1[:locusAncTimes.shape[0],:] = locusAncTimes
	locusTimes = np.array([row0,row1])
	return locusTimes, n, m

def load_data(args):
		# load coalescence times
	noCoals = (args.times == None)
	if not noCoals:
		times, n, m = load_times(args)
		if args.popFreq == None:
			x0 = n/(n+m)
		else:
			x0 = args.popFreq
	else:
		times = np.zeros((2,0,0))
		x0 = args.popFreq
	if x0 == None:
		currFreq = -1
	else:
		currFreq = x0

	# load ancient samples/genotype likelihoods
	if args.ancientSamps != None:
		ancientGLs = np.genfromtxt(args.ancientSamps,delimiter='\t')
	else:
		ancientGLs = np.zeros((0,4))

	if noCoals:
		tCutoff = np.max(ancientGLs[:,0])+1.0
	else:
		tCutoff = args.tCutoff

	epochs = np.arange(0.0,tCutoff)
	# loading population size trajectory
	if args.coal != None:
		Nepochs = np.genfromtxt(args.coal,skip_header=1,skip_footer=1)
		N = 0.5/np.genfromtxt(args.coal,skip_header=2)[2:-1]
		N = np.array(list(N)+[N[-1]])
		Ne = N[np.digitize(epochs)-1]
	else:
		Ne = args.N * np.ones(int(tCutoff))

	# load z tables
	z_bins,z_logcdf,z_logsf = load_normal_tables()

	# set up freq bins
	a=1
	b=a
	c = 1/(2*Ne[0])
	df = 100
	freqs = stats.beta.ppf(np.linspace(c,1-c,df),a,b)

	# load time bins (for defining selection epochs)
	timeBins = np.genfromtxt(args.timeBins)

	return timeBins,times,epochs,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,noCoals,currFreq

def likelihood_wrapper(theta,timeBins,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,gens,noCoals,currFreq):
    S = theta
    Sprime = np.concatenate((S,[0.0]))
    if np.any(np.abs(Sprime) > 0.1):
        return np.inf

    sel = Sprime[np.digitize(epochs,timeBins,right=False)-1]

    tShape = times.shape
    if tShape[2] == 0:
    	t = np.zeros((2,0))
    	importanceSampling = False
    elif tShape[2] == 1:
    	t = times[:,:,0]
    	importanceSampling = False
    else:
    	importanceSampling = True

    if importanceSampling:
    	M = tShape[2]
    	loglrs = np.zeros(M)
    	for i in range(M):
    		betaMat = backward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals,currFreq=currFreq)
    		logl = logsumexp(betaMat[-2,:])
    		logl0 = proposal_density(times[:,:,i],epochs,N)
    		loglrs[i] = logl-logl0
    	logl = -1 * (-np.log(M) + logsumexp(loglrs))
    else:
    	betaMat = backward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals,currFreq=currFreq)
    	logl = -logsumexp(betaMat[-2,:])
    #print(logl,S)
    return logl

def out(args,epochs,freqs,post):
	np.save(args.out+'.epochs',epochs)
	np.save(args.out+'.freqs',epochs)
	np.save(args.out+'.post',post)
	return

def traj_wrapper(theta,timeBins,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,gens,noCoals,currFreq):
    S = theta
    Sprime = np.concatenate((S,[0.0]))
    if np.any(np.abs(Sprime) > 0.1):
        return np.inf

    sel = Sprime[np.digitize(epochs,timeBins,right=False)-1]
    T = len(epochs)
    F = len(freqs)
    tShape = times.shape
    if tShape[2] == 0:
    	t = np.zeros((2,0))
    	importanceSampling = False
    elif tShape[2] == 1:
    	t = times[:,:,0]
    	importanceSampling = False
    else:
    	importanceSampling = True

    if importanceSampling:
    	M = tShape[2]
    	loglrs = np.zeros(M)
    	postBySamples = np.zeros((F,T-1,M))
    	for i in range(M):
    		betaMat = backward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals,currFreq=currFreq)
    		alphaMat = forward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals)
    		logl = logsumexp(betaMat[-2,:])
    		logl0 = proposal_density(times[:,:,i],epochs,N)
    		loglrs[i] = logl-logl0
    		postBySamples[:,:,i] = (alphaMat[1:,:] + betaMat[:-1,:]).transpose()
    	post = logsumexp(loglrs + postBySamples,axis=2)
    	post -= logsumexp(post,axis=0)

    else:
    	post = np.zeros((F,T))
    	betaMat = backward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals,currFreq=currFreq)
    	alphaMat = forward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,noCoals=noCoals)
    	post = (alphaMat[1:,:] + betaMat[:-1,:]).transpose()
    	post -= logsumexp(post,axis=0)
    return post

if __name__ == "__main__":
	args = parse_args()
	if args.times == None and args.ancientSamps == None:
		print('You need to supply coalescence times (--times) and/or ancient samples (--ancientSamps)')
	
	print()
	print('Loading data and initializing model...')

	# load data and set up model
	timeBins,times,epochs,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,noCoals,currFreq = load_data(args)

	# optimize over selection parameters
	T = len(timeBins)
	S0 = 0.0 * np.ones(T-1)
	opts = {'xatol':1e-4}

	if T == 2:
		Simplex = np.reshape(np.array([-0.05,0.05]),(2,1))
	elif T > 2:
		Simplex = np.zeros((T,T-1))
		for i in range(Simplex.shape[1]):
			Simplex[i,:] = -0.01
			Simplex[i,i] = 0.01
		Simplex[-1,:] = 0.01
	else:
		raise ValueError

	#bounds = tuple([(-0.05,0.05) for i in range(T-1)])
	opts['initial_simplex']=Simplex
	    
	#for tup in product(*[[-1,1] for i in range(3)]):
	logL0 = likelihood_wrapper(S0,timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,epochs,noCoals,currFreq)

	print('Optimizing likelihood surface using Nelder-Mead...')
	if times.shape[2] > 1:
		print('\t(Importance sampling with M = %d Relate samples)'%(times.shape[2]))
		print()
	minargs = (timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,epochs,noCoals,currFreq)
	res = minimize(likelihood_wrapper,
	         S0,
	         args=minargs,
	         options=opts,
	         #bounds=bounds,
	        method='Nelder-Mead')

	S = res.x
	L = res.fun
	#Hinv = np.linalg.inv(res.hess)
	#se = np.sqrt(np.diag(Hinv))

	print('#'*10)
	print()
	print('logLR: %.2f'%(-res.fun+logL0))
	print()
	print('epoch\tselection')
	for s,t,u in zip(S,timeBins[:-1],timeBins[1:]):
		print('%d-%d\t%.3f'%(t,u,s))
	

	# infer trajectory @ MLE of selection parameter
	post = traj_wrapper(res.x,timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,epochs,noCoals,currFreq)
	
	if args.out != None:
		out(args,epochs,freqs,post)
	else:
		traj = freqs[np.argmax(post,axis=0)][:100]
		print()
		print(traj)



