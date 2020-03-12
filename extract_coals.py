import argparse
from argparse import ArgumentParser
import warnings
import numpy as np
import glob
from scipy.stats import chi2
import scipy.stats as stats
from scipy.special import logsumexp
from scipy.optimize import minimize
import progressbar
import sys
from numba import njit

from Bio import Phylo
from io import StringIO
import tree_utils

def locus_parse_coal_times(args):
	bedFile = args.tree
	derivedAllele = args.derivedAllele
	posn = args.posn
	sitesFile = args.haps
	outFile = args.out
	timeScale = args.timeScale
	burnin = args.burnin
	thin = args.thin
	debug = args.debug

	indLists = tree_utils._derived_carriers_from_haps(sitesFile,posn)
	derInds = indLists[0]
	ancInds = indLists[1]
	ancHap = indLists[2]

	n = len(derInds)
	m = len(ancInds)
	
	f = open(bedFile,'r')
	lines = f.readlines()
	lines = [line for line in lines if line[0] != '#' and line[0] != 'R' and line[0] != 'N'][burnin::thin]
	numImportanceSamples = len(lines)

	derTimesList = []
	ancTimesList = []

	for (k,line) in enumerate(lines):
		print(k)
		nwk = line.rstrip().split()[-1]
		derTree =  Phylo.read(StringIO(nwk),'newick')
		ancTree = Phylo.read(StringIO(nwk),'newick')
		mixTree = Phylo.read(StringIO(nwk),'newick')

		derTimes,ancTimes,mixTimes = tree_utils._get_times_all_classes(derTree,ancTree,mixTree,
							derInds,ancInds,ancHap,n,m,sitesFile)
		derTimesList.append(derTimes)
		ancTimesList.append(ancTimes)

	#	if args.debug:
	#		bar.update(k+1)

	

	#if args.debug:
	#	bar.finish()
	times = -1 * np.ones((2,n+m,numImportanceSamples))
	for k in range(numImportanceSamples):
		times[0,:len(derTimesList[k]),k] = np.array(derTimesList[k])
		times[1,:len(ancTimesList[k]),k] = np.array(ancTimesList[k])
	return times

def _args_passed_to_locus(args):
	passed_args = args

	# reach into args and add additional attributes
	d = vars(passed_args)

	d['popFreq'] = 0.50

	d['posn'] = args.posn 
	d['derivedAllele'] = args.derivedAllele 
	return passed_args


def _args(super_parser,main=False):
	if not main:
		parser = super_parser.add_parser('locus_extract',description=
                'Parse/extract coalescence times in the derived & ancestral classes & compute LD statistics.')
	else:
		parser = super_parser
	# mandatory inputs:
	required = parser.add_argument_group('required arguments')
	required.add_argument('--tree',type=str,help='newick trees sampled by Relate at the SNP of interest (using SampleBranchLengths.sh)')
	required.add_argument('--haps',type=str,help='haps file used by Relate')
	required.add_argument('--posn',type=int,help='position of SNP of interest (must be polymorphic in sample!)')
	required.add_argument('--derivedAllele',type=str,help='character of derived allele (A/G/T/C)')
	required.add_argument('--out',type=str,help='prefix for outfiles (.der.npy, .anc.npy)')
	# options:
	parser.add_argument('-q','--quiet',action='store_true')
	parser.add_argument('-debug','--debug',action='store_true')

	
	parser.add_argument('-timeScale','--timeScale',type=float,help='Multiply the coal times \
						 	in bedFile by this factor to get in terms of generations; e.g. use \
						 	this on trees in units of 4N gens (--timeScale <4*N>)',default=1)
	parser.add_argument('-thin','--thin',type=int,default=1)
	parser.add_argument('-burnin','--burnin',type=int,default=0)	
	return parser

def _write_times_files(args,locusTimes):
	
	i0 = np.argmax(locusTimes[0,:,0] < 0.0) 
	i1 = np.argmax(locusTimes[1,:,0] < 0.0)
	a1 = locusTimes[0,:i0,:]
	a2 = locusTimes[1,:i1,:]
	a1 = a1.transpose()	
	a2 = a2.transpose()
	np.save(args.out+'.der.npy',a1)
	np.save(args.out+'.anc.npy',a2)
	return 	

def _parse_locus_stats(args):
	passed_args = _args_passed_to_locus(args)
	locusTimes = locus_parse_coal_times(passed_args)
	_write_times_files(args,locusTimes)
	return

def _main(args):	
	_parse_locus_stats(args)

if True:
        super_parser = argparse.ArgumentParser()
        parser = _args(super_parser,main=True)
        args = parser.parse_args()
        _main(args)
