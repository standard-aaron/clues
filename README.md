# adna_hmm

## Basic usage

`
python inference.py --ancientSamps example/exampleAncientSamps.txt --timeBins example/timeBins.txt --out example
`

where 

`exampleAncientSamps.txt` is a file listed ancient genotype likelihoods (set likelihoods to 0/-inf if hard-called), and

`timeBins.txt` is a file that denotes epochs (e.g. 0-50 gens before present) during which selection is allowed to differ from 0 during optimization. Note that you can set arbitrarily many time bins (e.g., 0,50,100,150...), but the estimator error will increase drastically, and optimization duration will also increase.

This will save three files: (1) `example.epochs.npy`, (2) `example.freqs.npy`, (3) `example.post.npy`. These files are discrete sets of (1) timepoints and (2) allele frequencies, as well as (3) the posterior distribution on allele frequency as a function of time. The posterior is saved in log space. All of these files can be loaded in python using `np.load()`

## Options

Firstly, `--ancientSamps` is actually not a required argument. At least one of `--ancientSamps` and `--times` must be specified. The option `--times` specifies coalescence times in the derived and ancestral classes (e.g. estimated using Relate). As an example, you can specify `--times example/example`, which points the program to `example/example.{der,anc}.npy`.

There is also an option to modify constant population size (`-N`) or population size changes (`--coal`). See the example folder for an example .coal file.
