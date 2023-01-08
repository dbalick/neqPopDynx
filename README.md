# neqPopDynx
_______________
_______________
Non-equilibrium Dynamix simulator 
\
Daniel J. Balick
_______________
Current version 1.5.0 \
Last edited: Jun 18, 2022 
_______________
>>>
This is a Wright-Fisher simulator for non-equilibrium population genetics that models the evolution of a number independent loci as they evolve through time-dependent changes in population size, strength of selection, and/or mutation rate.  Note that this is a work in progress, so please email dbalick@hms.harvard.edu or dbalick@gmail.com regarding any bugs or feature requests, in addition to questions and comments.  
\
The intended purpose is to output temporal data from an evolving population in the form of time-dependent trajectories for various observables of interest as they respond to simulated non-equilibrium population genetic phenomena.  The primary purpose is for comparison to predictive analytic models to evaluate accuracy and robustness to various scenarios and parameter regimes. It was written to assess predictions using a method that will be published shortly.  However, the simulator is relatively fast and flexible, so it can be used for a variety of purposes to produce time series output to model population genetics.   Output is produced (in the current iteration v1.5.1) in the form of the first five cumulants, non-central moments, and/or central moments of the allele frequency probability distribution (i.e., the site frequency spectrum for a single site).   L sites evolve independently and theiir frequencies are averaged to estimate these observables at regular time intervals.  There are a number of input options to parameterize diploid natural selectoin, mutation and back mutation rates of any level (e.g.,, recurrent mutations), and parameterized demographic scenarios corresponding to a time-dependent population size.
\
Burn-in is optional to track equilibration dynamics from an initial condtion.  The initial frequency is currently a delta function distribution at a specifiable allele count (the initial frequency will be dividded by the initial population size)
>>>
------------------------
>>**Requirements:** 
>>------------------------
>>Python 3 \
>>Numpy \
>>Scipy \
>>termcolor (for color-specific output to the terminal; this is installed automatically unless commented out) 
>>
------------------------
>>**Currently available files for download:**
>>------------------------
>>neqPopDynx_v1.5.0.py  (simulator script) \
>> bash_scripts_to_run_neqPopDynx_v1.5.zip  (contains example bash scripts)
>>>**Inside zip file:** 
>>>_Each example is a serial loop over parameters (add \& inside each loop to run in parallel)_\
>>>>run_equilib_N1e4_nPD.sh  (constant population size) \
>>>>run_exponential_Ni1e2_growth1e-3_nPD.sh  (exponential population size) \
>>>>run_bottleneck_Ni1e4_Nb1e2_nPD.sh  (square bottleneck in population size with re-expansion)\
>>>>run_oscillating_per1e4_Nmax1e4_Nmin1e3_nPD.sh  (oscillatory population size)\
_________________________

For now, please use these example scripts to learn how to run the simulator and test various styles of output.  
Please use 'neqPopDynx_v1.5.0.py -h' to display available command line opetions. 
Time-dependent selection coefficients are now working, but I have not yet posted an example script.
More details to come when I have more time.

