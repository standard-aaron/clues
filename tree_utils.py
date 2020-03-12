from Bio import Phylo
from io import StringIO
import numpy as np

def _coal_times(clades):
        # get number of leaf nodes

        [left,right] = clades
        lbl =  float(left.branch_length)
        rbl =  float(right.branch_length)

        #print lbl, rbl
        if len(left.clades) == 0 and len(right.clades) == 0:
            return [rbl]

        elif len(left.clades) == 0:
            right_times =  _coal_times(right.clades)
            return [lbl] + right_times

        elif len(right.clades) == 0:
            left_times =  _coal_times(left.clades)
            return [rbl] + left_times

        else:
            left_times =  _coal_times(left)
            right_times =  _coal_times(right)

            if lbl < rbl:
                return [lbl + left_times[0]] + left_times + right_times
            else:
                return [rbl + right_times[0]] + left_times + right_times

def _derived_carriers_from_haps(hapsFile,posn):
    f = open(hapsFile,'r')
    lines = f.readlines()
    for line in lines:
        posnLine = int(line.split()[2])
        if posnLine != posn:
            continue
        if posnLine == posn:
            alleles = ''.join(line.rstrip().split()[5:])
            hapsDer = [str(i) for i in range(len(alleles)) if alleles[i] == '1']
            hapsAnc = [str(i) for i in range(len(alleles)) if alleles[i] != '1'] 
            return [hapsDer,hapsAnc,[]]

def _get_times_all_classes(derTree,ancTree,mixTree,derInds,ancInds,ancHap,n,m,sitesFile,timeScale=1):
    
    indsToPrune = []
    #print(indsToPrune)
    if sitesFile == None:
        ### assume all individuals are fixed for the derived type!
        if ancHap != None:
            raise NotImplementedError
        else:
            derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
            ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
            mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))

    if ancHap == None:
        ancHap = []
    if n >= 2 and m >= 2:   
        for ind in set(ancInds + ancHap + indsToPrune):
            #print('der',ind)
            derTree.prune(ind)
        for ind in set(derInds + ancHap + indsToPrune):
            #print('anc',ind)
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        ancTimes = timeScale *np.sort(_coal_times(ancTree.clade.clades))
        mixTimes = timeScale *np.sort(_coal_times(mixTree.clade.clades))


    elif n == 1 and m >= 2:
        for ind in set(derInds + ancHap + indsToPrune):
            ancTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        derTimes = np.array([])
    
    elif n >= 2 and m == 1:
        for ind in set(ancInds + ancHap + indsToPrune):
            derTree.prune(ind)
        for ind in set(derInds[1:] + ancHap + indsToPrune):
            mixTree.prune(ind)
    
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))   
        mixTimes = timeScale * np.sort(_coal_times(mixTree.clade.clades))
        ancTimes = np.array([])

    elif n == 0 and m >= 2:
        Cder = [0]
        for ind in set(ancHap + indsToPrune):
            ancTree.prune(ind)
        ancTimes = timeScale * np.sort(_coal_times(ancTree.clade.clades))
        derTimes = np.array([])
        mixTimes = np.array([])

    elif n >= 2 and m == 0:
        Canc = [0]
        for ind in set(ancHap + indsToPrune):
            derTree.prune(ind)
        derTimes = timeScale * np.sort(_coal_times(derTree.clade.clades))
        ancTimes = np.array([])
        mixTimes = np.array([])
    return derTimes,ancTimes,mixTimes

