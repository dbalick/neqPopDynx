# neqPopDynx
_______________
_______________
Non-equilibrium Dynamix simulator
Daniel J. Balick
_______________
Current version 1.5.0
Last edited: Jun 18, 2022 
_______________
Please email dbalick@hms.harvard.edu or dbalick@gmail.com for comments, questions, or version notes


This is a Wright-Fisher simulator for non-equilibrium population genetics that models the evolution of a number independent loci as they evolve through time-dependent changes in population size, strength of selection, and/or mutation rate.  Note that this is a work in progress, so please email balick@hms.harvard.edu or dbalick@gmail.com regarding any bugs or feature requests, in addition to questions and comments.  

The intended purpose is to output temporal data from an evolving population in the form of time-dependent trajectories for various observables of interest as they respond to simulated non-equilibrium population genetic phenomena.  The primary purpose is for comparison to predictive analytic models to evaluate accuracy and robustness to various scenarios and parameter regimes. It was written to assess predictions using a method that will be published shortly.  However, the simulator is relatively fast and flexible, so it can be used for a variety of purposes to produce time series output to model population genetics.   Output is produced (in the current iteration v1.5.1) in the form of the first five cumulants, non-central moments, and/or central moments of the allele frequency probability distribution (i.e., the site frequency spectrum for a single site).   L sites evolve independently and theiir frequencies are averaged to estimate these observables at regular time intervals.  There are a number of input options to parameterize diploid natural selectoin, mutation and back mutation rates of any level (e.g.,, recurrent mutations), and parameterized demographic scenarios corresponding to a time-dependent population size.  
