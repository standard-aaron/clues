from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputPrefix',type=str)
parser.add_argument('figurePrefix',type=str)
parser.add_argument('--ext',type=str,default='pdf')
args = parser.parse_args()

epochs = np.load('%s.epochs.npy'%(args.inputPrefix))
freqs = np.load('%s.freqs.npy'%(args.inputPrefix))
logpost = np.load('%s.post.npy'%(args.inputPrefix))

f,ax = plt.subplots(1,1)
f.set_size_inches(20,10)

plt.pcolormesh(epochs[:-1],freqs,np.exp(logpost)[:,:])
plt.axis((0,800,0,1.0))
plt.ylabel('Allele frequency',fontsize=20)
plt.xlabel('Generations before present',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

cbar = plt.colorbar()
cbar.ax.set_ylabel('Posterior prob.\n\n',rotation=270,fontsize=20,labelpad=40)
cbar.ax.tick_params(labelsize=18) 

plt.savefig('%s.%s'%(args.figurePrefix,args.ext),format=args.ext)