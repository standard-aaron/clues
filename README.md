# `clues`

## Basic usage

`
python inference.py --times example/example --out example/out
`

where 

`example/example.{der,anc}.npy` are NumPy binary files containing derived/ancestral coalescence times. These are estimated using Relate [Speidel et al 2019]) and processed using the `extract_coals.py` script. 

## Tutorials/Wikis

Please visit the wiki pages found here (https://github.com/35ajstern/clues/wiki/Sampling-&-extracting-coalescence-times) to find tutorials of how to:

  1. Run `Relate`
      
      (https://github.com/35ajstern/clues/wiki/Sampling-&-extracting-coalescence-times#1-running-relate-and-samplebranchlengthssh)
      
  2. Prepare `clues` input files
  
      (https://github.com/35ajstern/clues/wiki/Sampling-&-extracting-coalescence-times#2-extracting-coalescence-times-using-extract_coalspy)
      
  3. Run `clues` & command-line options
  
      (https://github.com/35ajstern/clues/wiki/Command-line-options-for-CLUES)
      
  3. Plot the output of `clues` (ie, plot frequency trajectories)
  
      (https://github.com/35ajstern/clues/wiki/CLUES-output-&-Plotting-trajectories)

#### Previous implementation (`clues-v0`)

To find the previous version of `clues`, which uses ARGweaver output (Rasmussen et al, 2014; Hubisz, et al, 2019; http://compgen.cshl.edu/ARGweaver/doc/argweaver-d-manual.html), please go to https://github.com/35ajstern/clues-v0. We are no longer maintaining `clues-v0`
