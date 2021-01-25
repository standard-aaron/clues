import numpy as np
from hmm_utils import forward_algorithm
from hmm_utils import backward_algorithm
from hmm_utils import proposal_density
from scipy.special import logsumexp
import scipy.stats as stats
from scipy.optimize import minimize
import argparse
import gzip

def parse_clues(filename,args):
    with gzip.open(filename, 'rb') as fp:
        try:
            #parse file
            data = fp.read()
        except OSError:
            with open(filename, 'rb') as fp:
                try:
                    #parse file
                    data = fp.read()
                except OSError:
                    print('Error: Unable to open ' + filename)
                    exit(1)
           
        #get #mutations and #sampled trees per mutation
        filepos = 0
        num_muts, num_sampled_trees_per_mut = np.frombuffer(data[slice(filepos, filepos+8, 1)], dtype = np.int32)
        #print(num_muts, num_sampled_trees_per_mut)

        filepos += 8
        #iterate over mutations
        for m in range(0,num_muts):
            bp = np.frombuffer(data[slice(filepos, filepos+4, 1)], dtype = np.int32)
            filepos += 4
            anc, der = np.frombuffer(data[slice(filepos, filepos+2, 1)], dtype = 'c')
            filepos += 2
            daf, n = np.frombuffer(data[slice(filepos, filepos+8, 1)], dtype = np.int32)
            filepos += 8
            #print("BP: %d, anc: %s, der %s, DAF: %d, n: %d" % (bp, str(anc), str(der), daf, n))
            
            
            if daf >= n-1:
            	anctimes = np.empty((num_sampled_trees_per_mut,0))
            else:
                num_anctimes = 4*(n-daf-1)*num_sampled_trees_per_mut
                anctimes     = np.reshape(np.frombuffer(data[slice(filepos, filepos+num_anctimes, 1)], dtype = np.float32), (num_sampled_trees_per_mut, n-daf-1))
                filepos     += num_anctimes
            
            if daf <= 1:
            	dertimes = np.empty((num_sampled_trees_per_mut,0))
            else:
                num_dertimes = 4*(daf-1)*num_sampled_trees_per_mut
                dertimes     = np.reshape(np.frombuffer(data[slice(filepos, filepos+num_dertimes, 1)], dtype = np.float32), (num_sampled_trees_per_mut, daf-1))
                filepos     += num_dertimes
            
            if (args.A1 is not None) and (args.A1 != der.decode('ascii')):
                tmp = np.copy(dertimes)
                dertimes = np.copy(anctimes)
                anctimes = tmp 
		
    return dertimes,anctimes

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--times',type=str,
		help='Should refer to files <times>.{{der,anc}}.npy (exclude prefix .{{der,anc}}.npy)',
		default=None)
	parser.add_argument('--popFreq',type=float,default=None)
	parser.add_argument('-q','--quiet',action='store_true')
	parser.add_argument('-o','--output',dest='outFile',type=str,default=None)

	parser.add_argument('--ancientSamps',type=str,default=None)
	parser.add_argument('--ancientHaps',type=str,default=None)
	parser.add_argument('--out',type=str,default=None)

	parser.add_argument('-N','--N',type=float,default=10**4)
	parser.add_argument('-coal','--coal',type=str,default=None,help='path to Relate .coal file. Negates --N option.')
	parser.add_argument('--dom',type=float,default=0.5,help='dominance coefficient')
	parser.add_argument('--changePts',type=str,default=None,help='File listing times of ancestry changepoints.')

	# adv options
	parser.add_argument('--A1',type=str,default=None)
	parser.add_argument('--stepsize',type=float,default=1e-5)
	parser.add_argument('--thresh',type=float,default=1e-4)	
	parser.add_argument('-thin','--thin',type=int,default=1)
	parser.add_argument('-burnin','--burnin',type=int,default=0)
	parser.add_argument('--tCutoff',type=float,default=1000)
	parser.add_argument('--timeBins',type=str,default=None)
	parser.add_argument('--sMax',type=float,default=0.1)
	parser.add_argument('--tSkip',type=int,default=1)
	parser.add_argument('--df',type=int,default=150)
	parser.add_argument('--betaParam',type=float,default=0.5)
	parser.add_argument('--lik',action='store_true',help='saves likelihood function, used by PALM. **Only compatible when a single selection coefficient is estimated.**')
	parser.add_argument('--w',type=float,default=0.01,help='width used to estimate likelihood function (--lik)')
	return parser.parse_args()



def load_normal_tables():
    # read in global Phi(z) lookups
    z_bins = np.genfromtxt('utils/z_bins.txt')
    z_logcdf = np.genfromtxt('utils/z_logcdf.txt')
    z_logsf = np.genfromtxt('utils/z_logsf.txt')
    return z_bins,z_logcdf,z_logsf

def load_times(args):
	locusDerTimes,locusAncTimes = parse_clues(args.times+'.timeb',args)
	print(locusDerTimes.shape,locusAncTimes.shape)	
	if locusDerTimes.ndim == 0 or locusAncTimes.ndim == 0:
		raise ValueError
	#if np.prod(locusDerTimes.shape) == 0 or np.prod(locusAncTimes.shape) == 0:
	#	raise ValueError
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
		ancientGLs = np.genfromtxt(args.ancientSamps,delimiter=' ')
	else:
		ancientGLs = np.zeros((0,4))

	# load ancient haploid genotype likelihoods
	if args.ancientHaps != None:
		ancientHapGLs = np.genfromtxt(args.ancientHaps,delimiter=' ')
	else:
		ancientHapGLs = np.zeros((0,3))

	if noCoals:
		try:
			tCutoff = np.max(ancientGLs[:,0])+1.0
		except:
			tCutoff = np.max(ancientHapGLs[:,0])+1.0
	else:
		tCutoff = args.tCutoff

	epochs = np.arange(0.0,tCutoff,int(args.tSkip))
	# loading population size trajectory
	if args.coal != None:
		Nepochs = np.genfromtxt(args.coal,skip_header=1,skip_footer=1)
		N = 0.5/np.genfromtxt(args.coal,skip_header=2)[2:-1]
		N = np.array(list(N)+[N[-1]])
		Ne = N[np.digitize(epochs,Nepochs)-1]
	else:
		Ne = args.N * np.ones(int(tCutoff))

	# load z tables
	z_bins,z_logcdf,z_logsf = load_normal_tables()

	# set up freq bins
	a=args.betaParam
	b=a
	c = 1/(2*np.min([Ne[0],100000]))
	df = args.df 
	freqs = stats.beta.ppf(np.linspace(c,1-c,df),a,b)
	# load time bins (for defining selection epochs)
	if args.timeBins != None:
		timeBins = np.genfromtxt(args.timeBins)
	else:
		timeBins = np.array([0.0,tCutoff])


	if args.changePts != None:
		changePts = np.genfromtxt(args.changePts)
	else:
		changePts = np.array([])

	return timeBins,times,epochs,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,noCoals,currFreq,args.dom,changePts

def likelihood_wrapper(theta,timeBins,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,gens,noCoals,currFreq,h,sMax,changePts):
    S = theta
    Sprime = np.concatenate((S,[0.0]))
    if np.any(np.abs(Sprime) > sMax):
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
    		betaMat = backward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,currFreq=currFreq,h=h)
    		logl = logsumexp(betaMat[-2,:])
    		logl0 = proposal_density(times[:,:,i],epochs,N)
    		loglrs[i] = logl-logl0
    	logl = -1 * (-np.log(M) + logsumexp(loglrs))
    else:
    	betaMat = backward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,currFreq=currFreq,h=h)
    	logl = -logsumexp(betaMat[-2,:])
    #print(logl,S)
    return logl

def out(args,epochs,freqs,post):
	np.save(args.out+'.epochs',epochs)
	np.save(args.out+'.freqs',freqs)
	np.save(args.out+'.post',post)
	return

def traj_wrapper(theta,timeBins,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,gens,noCoals,currFreq,h,sMax,changePts):
    S = theta
    Sprime = np.concatenate((S,[0.0]))
    if np.any(np.abs(Sprime) > sMax):
        print('WARNING: selection coefficient exceeds bounds. Maybe change --sMax?')
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
    		betaMat = backward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,currFreq=currFreq,h=h)
    		alphaMat = forward_algorithm(sel,times[:,:,i],epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,h=h)
    		logl = logsumexp(betaMat[-2,:])
    		logl0 = proposal_density(times[:,:,i],epochs,N)
    		loglrs[i] = logl-logl0
    		postBySamples[:,:,i] = (alphaMat[1:,:] + betaMat[:-1,:]).transpose()
    	post = logsumexp(loglrs + postBySamples,axis=2)
    	post -= logsumexp(post,axis=0)

    else:
    	post = np.zeros((F,T))
    	betaMat = backward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,currFreq=currFreq,h=h)
    	alphaMat = forward_algorithm(sel,t,epochs,N,freqs,z_bins,z_logcdf,z_logsf,ancGLs,ancHapGLs,changePts,noCoals=noCoals,h=h)
    	post = (alphaMat[1:,:] + betaMat[:-1,:]).transpose()
    	post -= logsumexp(post,axis=0)
    return post

if __name__ == "__main__":
	args = parse_args()
	if args.times == None and args.ancientSamps == None and args.ancientHaps == None:
		print('You need to supply coalescence times (--times) and/or ancient samples (--ancientSamps) and/or ancient haploid samples (--ancientHaps)')
	
	print()
	print('Loading data and initializing model...')

	# load data and set up model
	sMax = args.sMax	
	timeBins,times,epochs,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,noCoals,currFreq,h,changePts = load_data(args)

	Ne *= 1/2
	noCoals = int(noCoals)

	# optimize over selection parameters
	T = len(timeBins)
	S0 = 0.0 * np.ones(T-1)
	opts = {'xatol':1e-4}

	if args.lik and T > 2:
		print('ERROR: Option --lik incompatible with estimating >1 selection coefficient.')
		raise ValueError

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
	logL0 = -likelihood_wrapper(S0,timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,epochs,noCoals,currFreq,h,sMax,changePts)

	print('Optimizing likelihood surface using Nelder-Mead...')
	if times.shape[2] > 1:
		print('\t(Importance sampling with M = %d Relate samples)'%(times.shape[2]))
		print()
	minargs = (timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,epochs,noCoals,currFreq,h,sMax,changePts)
	

	res = minimize(likelihood_wrapper,
		         S0,
		         args=minargs,
		         options=opts,
		         #bounds=bounds,
		        method='Nelder-Mead')

	S = res.x
	L = res.fun
	
	if args.lik:
		print('Fitting likelihood function...')
		print()
		Svec = np.linspace(S[0]-args.w,S[0]+args.w,20)
		Lvec = []
		for s in Svec:
			l = likelihood_wrapper(np.array([s]),timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,epochs,noCoals,currFreq,h,sMax,changePts)
			Lvec.append(l)
		Lvec = np.array(Lvec)
		S = np.array([Svec[np.argmax(-Lvec)]])
		L = np.max(-Lvec)
		p = np.polyfit(Svec,-Lvec,deg=2)
		if args.out != None:
			print(p)
			np.save(args.out+'.quad_fit.npy',p)


	print('#'*10)
	print()
	print('logLR: %.4f'%(L-logL0))
	print()
	print('MLE:')
	print('========')
	print('epoch\tselection')
	for s,t,u in zip(S,timeBins[:-1],timeBins[1:]):
		print('%d-%d\t%.7f'%(t,u,s))


	# infer trajectory @ MLE of selection parameter
	print(noCoals)

	post = traj_wrapper(S,timeBins,Ne,freqs,z_bins,z_logcdf,z_logsf,ancientGLs,ancientHapGLs,epochs,noCoals,currFreq,h,sMax,changePts)
	
	if args.out != None:
		out(args,epochs,freqs,post)
	else:
		print()
		print('Trajectory:')
		print('=============')
		print('gens_bp\tfreq')
		for i in range(0,int(timeBins[-1]/args.tSkip),int(timeBins[-1]//(50*args.tSkip))):
			if i*args.tSkip > epochs[-1]:
				break
			print(i*args.tSkip,np.sum(freqs * np.exp(post[:,i])))
		print()
		print('Finished.')
		print()


