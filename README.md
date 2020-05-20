# `clues`

## Basic usage

`
python inference.py --times example/example --out example/out
`

where 

`example/example.palm` is a binary file containing derived/ancestral coalescence times for the SNP of interest. These are estimated using [Relate v1.1](https://myersgroup.github.io/relate/) [[Speidel, *et al.* Nat. Gen. 2019]](https://www.nature.com/articles/s41588-019-0484-x) and processed using the `extract_coals.py` script. 

If you use this program, please cite:
  
  [Stern, *et al.* Plos Gen. (2019)](https://journals.plos.org/plosgenetics/article/metrics?id=10.1371/journal.pgen.1008384) (doi: 10.1371/journal.pgen.1008384)

## Tutorials/Wikis

Please visit the [Wiki pages](https://github.com/35ajstern/clues/wiki/Sampling-&-extracting-coalescence-times) to find tutorials of how to:

  1. [Run `Relate`](https://github.com/35ajstern/clues/wiki/Sampling-&-extracting-coalescence-times#1-running-relate-and-samplebranchlengthssh)
      
  2. [Run `clues` & command-line options (incl. incorporating aDNA samples)](https://github.com/35ajstern/clues/wiki/Command-line-options-for-CLUES)
      
  3. [Plot the output of `clues` (ie, plot frequency trajectories)](https://github.com/35ajstern/clues/wiki/CLUES-output-&-Plotting-trajectories)
      
#### Installation/dependencies

`clues` has developed for python 3; use python 2 at your own risk (visit the Issues section for tips on using python 2).

The programs require the following dependencies, which can be installed using conda/pip: `numba`, `progressbar`, `biopython`

#### Previous implementation (`clues-v0`)

To find the previous version of `clues`, which uses ARGweaver output (Rasmussen et al, 2014; Hubisz, et al, 2019; [docs here](http://compgen.cshl.edu/ARGweaver/doc/argweaver-d-manual.html)), please go to https://github.com/35ajstern/clues-v0. We are no longer maintaining `clues-v0`
