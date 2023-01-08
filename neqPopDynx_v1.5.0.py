#################################################
#                                               #
#             ** neqPopDynx **                  #
#      Non-equilibrium Population Dynamics      #
#         Wright-Fisher Simulator               #
#          for Transient dynamics               #
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           #
#    - general diploid selection                #
#    - flexible demography                      #
#    - temporal output                          #
#                                               #
#        ----- Version 1.5.0 -----              #
#                                               #
#    Daniel J. Balick                           #
#    Started scripting: Mar 15, 2022            #
#    Last edited:       Jun 18, 2022            #
#                                               #
#################################################

#neqPopDynx_version="1.5.0"
from __future__ import division, print_function
version_label = "v1.5.0"


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Please email dbalick@hms.harvard.edu or dbalick@gmail.com
#   for comments, questions, or version notes
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import scipy as sp
import scipy.stats as spstats
import sys, os
import optparse, itertools
#import itertools
import warnings
from numpy.random import poisson
from numpy.random import binomial
from numpy.random import shuffle
from numpy.random import choice
import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
import bisect
from bisect import bisect
import time
import subprocess
import sys
####  Install termcolor if not already
try:
    from termcolor import cprint
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'termcolor'])
finally:
    from termcolor import cprint

####    GLOBALS
starting_directory = os.getcwd()
start_time = time.time()

###################################################################################################

#______________________________________________________________
#______________________________________________________________
#
#                   UTILITY FUNCITONS
#______________________________________________________________
#______________________________________________________________

#### this makes sure a directory exists
def assure_path_exists(directory_temp):
    if os.path.exists(directory_temp):
        return
    else:
        os.makedirs(directory_temp)
    return

#______________________________________________________________
#______________________________________________________________
#
#                   PRINT FUNCITONS
#______________________________________________________________
#______________________________________________________________


####  May want to use print files with this label in the future:
#    filename_prefix = "nPD"+version_label

###  NOTE: THIS IS AN UPDATED print_moments_v3
def print_moments(shet, shom, mu, mub, L, initpopsize, demography, ancestry, growth_rate, BNpopsize, BNstart, BNduration, osc_period, burnin, generations, initcond, initcounts, printgen, run_number, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, finalpopsize, record_moments, record_central_moments, record_cumulants, which_selection, selection_parameter, later_s, folder_label):
    
    if folder_label!='-999':
        assure_path_exists(folder_label)
        os.chdir(folder_label)
    
    ####  May want to use this label in the future:
    #    filename_prefix = "nPD"+version_label
    
    initfreq=initcounts/initpopsize
    
    initpopsize=int(initpopsize)
    BNpopsize=int(BNpopsize)
    BNstart=int(BNstart)
    BNduration=int(BNduration)
    generations=int(generations)
    run_number=int(run_number)
    burnin=int(burnin)
    L=int(L)
    record_gen=record_gen.astype(np.int32)
    
#    file_location = "neqPopDynx_output/"+demography+"/init_condition_"+initcond+"_"+str(initfreq)+"/"
#    assure_path_exists(file_location)
#    os.chdir(file_location)
    if which_selection!='constant':
        which_selection_label="time_dependent_selection_"+which_selection
        which_selection_dir=which_selection_label+"/"
        if which_selection=='exp' and later_s==-999 and selection_parameter!=-999:
            which_selection_filename="_SofT"+which_selection+"_SelGrowth"+str(selection_parameter)
        elif which_selection=='step' and selection_parameter!=-999 and later_s!=-999:
            which_selection_filename="_SofT"+which_selection+"_SelChangeTo"+str(later_s)+"_NewSelGen"+str(selection_parameter)
        elif which_selection=='flip' and later_s==-999 and selection_parameter!=-999:
            which_selection_filename="_SofT"+which_selection+"_SelSignFlipGen"+str(selection_parameter)
    else:
        which_selection_label=""
        which_selection_dir=""
        which_selection_filename=""

    if (initpopsize/2)%10==0 and np.log10(initpopsize/2)==round(np.log10(initpopsize/2)):
        if demography=="oscillating":
            logN = "2Nmx2e"+str(int(np.log10(initpopsize/2)))
        else:
            logN = "2N2e"+str(int(np.log10(initpopsize/2)))
        demog_prefix=demography+"_"+logN
#        assure_path_exists(logN)
#        os.chdir(logN)
    else:
        if demography=="oscillating":
            demog_prefix = demography+"_2Nmx"+str(initpopsize)
        else:
            demog_prefix = demography+"_2N"+str(initpopsize)
#        assure_path_exists(str(initpopsize))
#        os.chdir(str(initpopsize))

    file_location = "neqPopDynx_"+version_label+"_output/"+which_selection_dir+demography+"/initpopsize_2N_"+str(initpopsize)+"_init_condition_"+initcond+"_"+str(initfreq)+"/"
    assure_path_exists(file_location)
    os.chdir(file_location)

    if burnin:
        burnpath = "with_burnin/"
        assure_path_exists(burnpath)
        os.chdir(burnpath)
    
    if run_number>0:
        runprefix = "_runtime"+str(generations)+"_run"+str(run_number)
    else:
        cprint("WARNING: NO SEED OR RUN NUMBER SPECIFIED!\nAnnotating with time stamp.",'red')
        runprefix = "_runtime"+str(generations)+"_time"+str(time.time())
        assure_path_exists("time_labeled_runs")
        os.chdir("time_labeled_runs")

    if demography=="exponential":
        demog_prefix=demog_prefix+"_growth"+str(growth_rate)
    elif demography=="tennessen" or demography=="supertennessen":
        demog_prefix=demog_prefix+"_"+ancestry
        if demography=="supertennessen":
            demog_prefix=demog_prefix+"_growth"+str(growth_rate)
    elif demography=="bottleneck":
        if (BNpopsize/2)%10==0 and np.log10(BNpopsize/2)==round(np.log10(BNpopsize/2)):
            logNB=int(np.log10(BNpopsize/2))
            demog_prefix=demog_prefix+"_2Nb2e"+str(logNB)+"_start"+str(BNstart)+"_Tb"+str(BNduration)
        else:
            demog_prefix=demog_prefix+"_2NB"+str(BNpopsize)+"_start"+str(BNstart)+"_TB"+str(BNduration)
    elif demography=="oscillating":
        if (BNpopsize/2)%10==0 and np.log10(BNpopsize/2)==round(np.log10(BNpopsize/2)):
            logNB=int(np.log10(BNpopsize/2))
            demog_prefix=demog_prefix+"_2Nmn2e"+str(logNB)+"_prd"+str(osc_period)
        else:
            demog_prefix=demog_prefix+"_2Nmn"+str(BNpopsize)+"_prd"+str(osc_period)

    logmu = str(np.log10(mu))
    if L%10!=0:
        logL = "_L"+str(L)
    else:
        logL = "_L1e"+str(np.log10(L))

    if mu>0:
        if mu%10!=0:
            logmu = "_mu"+str(mu)
        else:
            logmu = "_mu1e"+str(np.log10(mu))

    if mub>0:
        if mub%10!=0:
            logmu = logmu+"_mub"+str(mub)
        else:
            logmu = logmu+"_mub1e"+str(np.log10(mub))

    h_temp = shet/shom
    if h_temp < 0 or h_temp > 1 or np.isnan(h_temp):
        param_label = logmu+logL+"_shet"+str(shet)+"_shom"+str(shom)
    else:
        hlabel=str(shet/shom)
        hval = shet/shom
        sval=shom
        param_label = logmu+logL+"_h"+str(hval)+"_s"+str(sval)
#####  REMOVED THIS (NO LONGER CHANGES EVEN VALUES TO LOG10 STYLE)
#    else:
#        hlabel=str(shet/shom)
#        hval = shet/shom
#        if np.log10(-shom)%1==0:
#            slabel=str(np.log10(-shom))
#            param_label = logmu+logL+"_h"+hlabel+"_s-1e"+slabel
#        else:
#            param_label = logmu+logL+"_h"+hlabel+"_s"+str(shom)

    initcond_prefix = "_"+initcond+"_initp"+str(initfreq)

    full_filename = demog_prefix+param_label+initcond_prefix+which_selection_filename+runprefix+".tsv"

    column_labels = "demog\t2N(t=0)\t2N(final)\tL\tmu\tmub\tshet\tshom\tgeneration"
    if which_selection=='exp':
        column_labels="type_of_time_dependent_selection\ts_growth_rate\t"+column_labels
    elif which_selection=='flip':
        column_labels="type_of_time_dependent_selection\tgeneration_s_flips_sign\t"+column_labels
    elif which_selection=='step':
        column_labels="type_of_time_dependent_selection\tgeneration_s_changes\tshet_changes_to\t"+column_labels
    if record_moments:
        column_labels=column_labels+"\tE[p]\tE[p^2]\tE[p^3]\tE[p^4]\tE[p^5]"
    if record_central_moments:
        column_labels=column_labels+"\tVar[p]\tSkew[p]\tKurt[p]\t5thCentMom[p]"
    if record_cumulants:
        column_labels=column_labels+"\tCumulant1\tCumulant2\tCumulant3\tCumulant4\tCumulant5"
    column_labels=column_labels+"\tfixations\n"

    nPD_output_file = open(full_filename, "w")
    nPD_output_file.write(column_labels)
    for i in range(len(mean_freq)):
        print_this_stuff =demography+"\t"+str(initpopsize)+"\t"+str(finalpopsize)+"\t"+str(L)+"\t"+str(mu)+"\t"+str(mub)+"\t"+str(shet)+"\t"+str(shom)+"\t"+str(record_gen[i])
        if (which_selection=='exp' or which_selection=='flip') and later_s==-999 and selection_parameter!=-999:
            print_this_stuff=which_selection+"\t"+str(selection_parameter)+"\t"+print_this_stuff
        elif which_selection=='step' and later_s!=-999 and selection_parameter!=-999:
            print_this_stuff=which_selection+"\t"+str(selection_parameter)+"\t"+str(later_s)+"\t"+print_this_stuff
        if record_moments:
            print_this_stuff = print_this_stuff+"\t"+str(mean_freq[i])+"\t"+str(homozygosity[i])+"\t"+str(moment3[i])+"\t"+str(moment4[i])+"\t"+str(moment5[i])
        if record_central_moments:
            print_this_stuff = print_this_stuff+"\t"+str(var_freq[i])+"\t"+str(skew[i])+"\t"+str(kurt[i])+"\t"+str(central_moment5[i])
        if record_cumulants:
            print_this_stuff = print_this_stuff+"\t"+str(cumulant1[i])+"\t"+str(cumulant2[i])+"\t"+str(cumulant3[i])+"\t"+str(cumulant4[i])+"\t"+str(cumulant5[i])
        print_this_stuff = print_this_stuff+"\t"+str(fixed_sites[i])+"\n"
        nPD_output_file.write(print_this_stuff)
    nPD_output_file.close()

    if burnin:
        os.chdir("..")
    os.chdir("../../..")
    if which_selection!='constant':
        os.chdir("..")
    if folder_label!='-999':
        os.chdir("..")

    return







#############################################################################
######       ############################                   #################
########  ###  ##########################     DEMOGRAPHIC   #################
########  ####  #########################      HISTORY      #################
########  ###  ##########################       KERNEL      #################
######        ###########################                   #################
#########################################################################djb#

#______________________________________________________________
#______________________________________________________________
#
#                   DEMOGRAPHY CHOICE
#______________________________________________________________
#______________________________________________________________

def demography_resize(demog, generation, current_popsize, initial_popsize, growth_rate, BN_start_time, BN_duration, N_BN, ancestry, tennessen_second_epoch_popsize, osc_period):

    if demog == "equilib":
        pop_size_demog = initial_popsize
    elif demog == "dynamic":
        pop_size_demog = current_popsize
    elif demog == "exponential" and growth_rate != -999:
        pop_size_demog = exponential_growth(generation, initial_popsize, growth_rate)
    elif demog == "bottleneck" and BN_duration >0 and BN_start_time >= 0 and N_BN > 0:
        pop_size_demog = bottleneck(generation, initial_popsize, BN_start_time, BN_duration, N_BN)
    elif demog=="tennessen":
        pop_size_demog = tennessen(initial_popsize, generation, ancestry, tennessen_second_epoch_popsize)
    elif demog=="supertennessen":
        pop_size_demog = super_tennessen(initial_popsize, generation, "european", tennessen_second_epoch_popsize, growth_rate)
    elif demog=="oscillating" and (N_BN>0 and osc_period>0):
        Nmaxtemp= initial_popsize/2
        Nmintemp = N_BN/2
        pop_size_demog = oscillate_popsize(generation, Nmaxtemp, Nmintemp, osc_period)
    elif demog=="pseudoexp":
        pop_size_demog = pseudo_exponential(generation, initial_popsize, BN_start_time, growth_rate)
    elif demog=="toytennessen":
        pop_size_demog = toy_tennessen(generation, initial_popsize, BN_start_time, N_BN, growth_rate)
    else:
        cprint("\n\n\n\tERROR! THIS DEMOGRAPHY IS NOT YET DEFINED!\n\n\n",'red', attrs=['underline'])
        cprint("You specified the demography:"+demog+".\n",'red')
        cprint("Please choose from:\n'equilib', 'dynamic','exponential','bottleneck','oscillating','tennessen', or 'supertennessen'\n\n\nExiting...\n\n\n\n", 'red')
        sys.exit()
#        print("running equilibrium demography...")
#        pop_size_demog = demog_equilib(generation)

    return pop_size_demog


#def old_exponential_growth(generation, current_popsize, growth_rate):
#    exp_size = int(current_popsize*np.exp(growth_rate*generation))
#    return exp_size

def exponential_growth(generation, initpopsize, growth_rate):
    exp_size = int(initpopsize*np.exp(growth_rate*generation))
    return exp_size

def bottleneck(generation, initpopsize, BN_start_time, BN_duration, N_BN):
    if generation >= BN_start_time and generation < (BN_start_time+BN_duration):
        return N_BN
    else:
        return initpopsize

def tennessen(initpopsize, generation, ancestry, second_growth_init):
    if ancestry=="african":
        if generation < 5716:
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
            new_popsize = initpopsize
        elif generation>=5716 and generation<=5920:
            new_popsize = int(initpopsize*np.exp(0.0166*(generation-5716)))
    if ancestry=="european":
        if generation < 3880:
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
            new_popsize=initpopsize
        elif generation>=3880 and generation<5000:
                new_popsize=3722
        elif generation>=5000 and generation<5716:
            new_popsize=int(2064*np.exp(0.00307*(generation-5000)))
        elif generation>=5716:
            new_popsize=int(second_growth_init*np.exp(0.0195*(generation-5716)))
    return new_popsize

def super_tennessen(initpopsize, generation, ancestry, second_growth_init, second_growth_rate):
    if ancestry=="african":
        if generation < 5716:
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
            new_popsize = initpopsize
        elif generation>=5716 and generation<=5920:
            new_popsize = int(initpopsize*np.exp(0.0166*(generation-5716)))
    if ancestry=="european":
        if generation < 3880:
            # new_popsize=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
            new_popsize=initpopsizeE
        elif generation>=3880 and generation<5000:
                new_popsize=3722
        elif generation>=5000 and generation<5716:
            new_popsize=int(2064*np.exp(0.00307*(generation-5000)))
        elif generation>=5716:
            new_popsize=int(second_growth_init*np.exp(second_growth_rate*(generation-5716)))
    return new_popsize

def oscillate_popsize(generation, Nmax, Nmin, osc_period):
    if generation==0:
        osc_size= int(Nmax)
    else:
        osc_size = 2*int(2*(Nmax*Nmin)/(Nmax+Nmin + (-Nmax+Nmin)*np.cos(2*np.pi*generation/osc_period)))
    return osc_size


#################################################################################
##### THESE ARE UNTESTED AND NOT SET UP TO PRINT... FIX THIS BEFORE USING #######
#################################################################################
def pseudo_exponential(generation, initpopsize, exp_start, growth_rate):
    if generation < exp_start:
        exp_size = initpopsize
    elif generation >= exp_start:
        exp_size = int(initpopsize*np.exp(growth_rate*(generation-exp_start)))
    return exp_size

def toy_tennessen(generation, initpopsize, BN_start_time, N_BN, growth_rate):
    if generation < BN_start_time:
        new_size = initpopsize
    elif generation >= BN_start_time:
        new_size = int(N_BN*np.exp(growth_rate*(generation-BN_start_time)))
    return new_size



###########################################################################
#######################################################             #######
######         ###############    #####     ###########             #######
####    #############   ########    ####  #############  SELECTION  #######
######     #######        ######  #  ###  #############      &      #######
#########    #######   #########  ##  ##  #############    DRIFT    #######
###        ##################     ####     E ##########    KERNEL   #######
#######################################################             #######
#######################################################             #######
#######################################################################djb#


def selection_and_drift(counts, shet, shom, current_popsize, past_popsize, which_selection, selection_parameter, later_s, generation, userseed):
    
    if userseed != 0:
        np.random.seed(userseed)
    
    if which_selection != 'constant':
        shet = selection_time_dependent(which_selection, selection_parameter, shet, later_s, generation)
        shom=2*shet

    ##  define frequency from counts
    p = 1.0*counts/past_popsize
    ## THIS IS AN IMPORTANT BUG CHECK
    ##  check if any frequencies >1 or <0 (they shouldn't)
    max_frequency=1
    p = check_incorrect_counts(p ,max_frequency, "frequencies", generation, past_popsize, current_popsize, "selection_and_drift:frequencies")

    ## Apply selection
    expected_freq_numerator = (1+shom)*p*p + (1+shet)*p*(1-p)
    expected_freq_denom = (1+shom)*(p*p) + (1+shet)*2*p*(1-p) +(1-p)*(1-p)
    expected_freq = expected_freq_numerator/expected_freq_denom
 
    new_counts = np.array(np.random.binomial(current_popsize, expected_freq))



    return new_counts

######--------------------------------------------------------------------------------------------------
#   check fixations, fixation reversal (from back mutations), and for any errors in counts/frequencies
######--------------------------------------------------------------------------------------------------

def check_fixation(counts_temp, new_counts_temp, current_popsize_temp, fixations):
    is_fixed_before = np.where(counts_temp==current_popsize_temp)
    is_fixed_after = np.where(new_counts_temp==current_popsize_temp)
    is_fixed_before = np.unique(is_fixed_before[0])
    is_fixed_after = np.unique(is_fixed_after[0])
    for fix_new in is_fixed_after:
        if fix_new not in is_fixed_before:
            fixations[fix_new]+=1
    return fixations

def check_unfixation(counts, new_counts, current_popsize, fixations):
    is_fixed_before = np.where(counts==current_popsize)
    is_fixed_after = np.where(new_counts==current_popsize)
    is_fixed_before = np.unique(is_fixed_before[0])
    is_fixed_after = np.unique(is_fixed_after[0])
    for fix_new in is_fixed_before:
        if fix_new not in is_fixed_after:
            fixations[fix_new]-=1
    return fixations

def check_incorrect_counts(p, pmax, style, generation, past_popsize, current_popsize, where_in_code):
    if len(p[p>pmax])>0 or len(p[p<0])>0:
        cprint("\n\n\n\n\nWARNING: Allele "+style+" >"+str(pmax)+" or <0 encountered!\n",'red')
        cprint("This occured in "+where_in_code+" of nPD_"+version_label+"\n", 'yellow')
        print("generation, past popsize, current popsize = ",generation,", ", past_popsize,", ",current_popsize)
        ## Check for excessive frequencies p>1 or counts > past_popsize and report offenders
        unphysical_p = np.where(p>pmax)
        unphysical_p = np.unique(unphysical_p[0])
        print(style+" excceding "+str(pmax)+":\nIndexes ", unphysical_p,"\n"+style+": ")
        if len(unphysical_p)>0:
            highstring="[ "
            highstring+=str(p[1])+" "
            for jjj in unphysical_p[2:]:
                highstring+=(str(p[jjj])+" ")
                p[jjj]=pmax
            print(highstring+"]")
        else:
            print("none")
        print("\n\n")
        ##  Finished with unphysical frequencies/counts
        ## Check for negative frequencies/counts p<0
        # this should never happen
        negative_p = np.where(p<0)
        negative_p = np.unique(negative_p[0])
        print("negative "+style+":\nindex ", negative_p,"\n")
        print(style+": ")
        if len(negative_p)>0:
            for lll in negative_p:
                print(p[lll], " ")
        else:
            print("none")
        print("\n\n")
        ##  Finished with negative frequencies/counts
        ## MANUALLY CORRECT ALLELE FREQUENCIES p>pmax and p<0
        cprint("\n\n\n\n\nCAUTION: Manually correcting impossible "+style+" >"+str(pmax)+" and <0.\n",'red')
        p[p>pmax]=pmax
        print("\n\n"+style+" above "+str(pmax)+" have been manually set to "+str(pmax)+".\n\n")
        p[p<0]=0
        print("negative "+style+" have been manually set to zero.\n\n")
    return p


######--------------------------------------------------------------------------------------------------
#   Definitions of time-dependent selection: step function, exponential change, sign flip
######--------------------------------------------------------------------------------------------------

def selection_time_dependent(which_selection, selection_parameter, initial_s, later_s, generation):
    if which_selection=='step':
                                     # gen        time to change s    s(t=0)    s(t=inf)
        new_s = selection_heaviside(generation, selection_parameter, initial_s, later_s)
    elif which_selection=='exp':
                                # gen         growth rate        s(t=0)
        new_s = selection_exp(generation, selection_parameter, initial_s)
    elif which_selection=='flip':
                                    # gen         start signflip      s(t=0)
        new_s = selection_flipsign(generation, selection_parameter, initial_s)
    elif which_selection=='random':
                                    # gen            start noise       mean s    var s
        new_s = selection_with_noise(generation, selection_parameter, initial_s, later_s)
    return new_s

def selection_heaviside(gen, t_change_s, initial_s, later_s):
    if gen < t_change_s:
        return initial_s
    else:
        return later_s

def selection_exp(gen, grow_selection, initial_s):
    growing_s = int(initial_s*np.exp(grow_selection*gen))
    return growing_s

def selection_flipsign(gen, t_change_s, initial_s):
    if gen < t_change_s:
        return initial_s
    else:
        return -initial_s

### Gaussian noise
#  note that s can potentially change sign; to avoid this, use np.abs, np.sign on mean
#  and replace sign in 'return'. e.g., mean->abs(mean), return sign*abs(mean)
def selection_with_noise(gen, start_adding_noise, mean_s, var_s):
    if gen<start_adding_noise:
        return mean_s
    else:
        if selection_parameter!=-999:
            #### Specifiable variance using later_s
            noisy_s = np.random.normal(mean_s, np.sqrt(np.abs(var_s)))
        else:
            #### Gaussian approx to Poisson as a default
            noisy_s = np.random.normal(mean_s, np.sqrt(np.abs(mean_s)))
        return noisy_s


#############################################################################
##   ####   ##  #####  ######################################################
##  # ## #  ##  #####  #######                                             ##
##  ##  ##  ##  #####  #######     MUTATION KERNEL (with back mutations)   ##
##  ######  ##  #####  #######                                             ##
##  ######  ###       #######################################################
#########################################################################djb#

def mutation(counts, fixations, mu, mub, current_popsize, userseed):
    
    if userseed != 0:
        np.random.seed(userseed)
    
    new_mu = poisson(mu*(current_popsize - counts))
    if mub >0:
        ### IMPORTANT!  THERE WAS A BUG HERE:
        # new_mub = poisson(mub*(current_popsize - counts))
        new_mub = poisson(mub*counts)
        new_counts = counts + new_mu - new_mub
    else:
        new_counts = counts + new_mu

    ### Trim counts at unphysical frequency
    #       (can probably also do this with a cieling/floor)
    copy_counts=new_counts
    new_counts[new_counts<0]=0
    new_counts[new_counts>current_popsize]=current_popsize
    if len(np.where(new_counts<0)[0])>0 or len(np.where(new_counts>current_popsize)[0])>0:
        cprint("\n\tERROR!! Mutated counts >popsize or <0.",'red')
        cprint("Offending loci at site(s) <0: "+str(np.where(new_counts<0)[0])+" and >2N: "+str(np.where(new_counts>current_popsize)[0])+"\n",'red')
        if len(new_counts>10):
            print("corrected(last 10 gen)=",new_counts[-10:],"\noriginal(last 10)=",copy_counts[-10:],"\n")
        else:
            print("corrected=",new_counts,"\noriginal=",copy_counts,"\n")
        cprint("\n\n\nExiting...\n\n\n\n",'red')
        sys.exit()

    if len(np.where(new_counts>current_popsize)[0])>0 or len(np.where(new_counts<0)[0]>0):
        cprint("\n\nWARNING: PROBLEM CLIPPING MUTATIONS ABOVE/BELOW ALLOWED REGIME!\n\n",'red')

    return new_counts



##########################################################################################
######              ##########################    ###########                      #######
########   ###################################   ############       EVOLUTION      #######
########         #####    ####  ###      #####  #############          PER         #######
########   ############   ##  ###   ###   ###################       GENERATION     #######
######              ####    ######      ####   ##############                      #######
######################################################################################djb#


def evolve(counts, shet, shom, L, mu, mub, current_popsize, past_popsize, gen, maxgen, fixations, which_selection, selection_parameter, later_s, userseed):
    

    ##########################################
    #   selection and drift
    ##########################################
    selected_counts = selection_and_drift(counts, shet, shom, current_popsize, past_popsize, which_selection, selection_parameter, later_s, gen, userseed)
    
    #### check for fixation
    fixations = check_fixation(counts, selected_counts, current_popsize, fixations)
    ######  WRITE FUNCTION TO CHECK EXTINCTIONS??

    ##########################################
    #   mutation
    ##########################################
    mutated_new_counts = mutation(selected_counts, fixations, mu, mub, current_popsize, userseed)
    
    #### check for unfixations (back mutation of fixed alleles)
    fixations = check_unfixation(selected_counts, mutated_new_counts, current_popsize, fixations)

    return mutated_new_counts, fixations



#############################################################################
###      ####################################################################
######        ###############################################################
##########                                                        ###########
##############          COMPUTE STATS FOR OUTPUT                  ###########
##############                                                    ###########
#############################################################            ####
################################################################            #
#############################################################################


def compute_stats(counts, current_popsize, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, fixations, gen, record_moments, record_central_moments, record_cumulants):
    
    ## Bias correction
    Lminus1 = (len(counts)-1)
    ## Define frequency
    x = counts/current_popsize
    ## mean for central moments
    xbar = np.mean(x)
    
    rawmu1 = xbar
    if record_moments:
        rawmu2 = np.mean(x**2)
        rawmu3 = np.mean(x**3)
        rawmu4 = np.mean(x**4)
        rawmu5 = np.mean(x**5)
    

    if record_central_moments or record_cumulants:
        mu1 = xbar
        mu2 = np.sum((x-xbar)**2)/Lminus1
        mu3 = np.sum((x-xbar)**3)/Lminus1
        mu4 = np.sum((x-xbar)**4)/Lminus1
        mu5 = np.sum((x-xbar)**5)/Lminus1
        if record_cumulants:
            K1 = mu1
            K2 = mu2
            K3 = mu3
            K4 = mu4- 3*(mu2**2)
            K5 = mu5 - 10*(mu2*mu3)

    ###  THIS IS A BIG BUG CHECK TO CONFIRM THERE ARE NO FREQUENCIES ABOVE 1
    ###    (can happen with changing popsize if selection/drift are incorrectly phrased)
    if xbar>1 or len(x[x>1])>1:
        cprint("\n\n\n\n\nWARNING: Mean frequency > 1 encountered!\n",'red')
        print("Generation: ", gen)
        print("Population size =", current_popsize)
        print("Min, Max counts:",np.min(counts), np.max(counts))
        print("Mean = ",rawmu1)
        if record_moments:
            print("Hom = ",rawmu2)
            print("Third non-central moment = ",rawmu3)
            print("Fourth non-central moment = ",rawmu4)
        if record_central_moments or record_cumulants:
            print("Var = ",mu2)
            print("Skew (non-standardized) = ", mu3)
            if record_central_moments:
                print("Kurtosis (non-standardized) = ", mu4)
            if record_cumulants:
                print("Excess Kurtosis (non-standardized) = ", K4)
        check_incorrect_counts(counts, current_popsize, "counts", gen, "N/A", current_popsize, "stats:p>1flag")

    mean_freq = np.append(mean_freq, rawmu1)
    if record_moments:
        homozygosity = np.append(homozygosity, rawmu2)
        moment3 = np.append(moment3, rawmu3)
        moment4 = np.append(moment4, rawmu4)
        moment5 = np.append(moment5, rawmu4)

    if record_central_moments:
        var_freq = np.append(var_freq, mu2)
        skew = np.append(skew, mu3)
        kurt = np.append(kurt, mu4)
        central_moment5 = np.append(central_moment5, mu5)

    if record_cumulants:
        cumulant1 = np.append(cumulant1, K1)
        cumulant2 = np.append(cumulant2, K2)
        cumulant3 = np.append(cumulant3, K3)
        cumulant4 = np.append(cumulant4, K4)
        cumulant5 = np.append(cumulant5, K5)
    
    fixed_sites = np.append(fixed_sites, np.sum(fixations))
    record_gen = np.append(record_gen, gen)

    return mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen






########################################################################################################
###################      ######   ###########    ##########       #####     ######    ..................
######################   ######   ########   ###  ###########   ########     #####  ####################
######################    ####    ######   #######   ########   ########   #  ####  ####################
######################  #  #  ##  ######   #######   ########   ########   ##  ###  ####################
######################  ##   ###  ######             ########   ########   ###  ##  ####################
######################  ########  ######   #######   ########   ########   ####  #  ####################
######################  ########  ######   #######   ########   ########   #####    ####################
#..................     ########   ....    #######    .....       ....     ######     ##################
####################################################################################################djb#


def main(userseed, shet, shom, L, mu, mub, demography, ancestry, growth_rate, initpopsize, BNstart, BNduration, BNpopsize, generations, burnin, printgen, logprint, skipdemog, run_number, initcond, initcounts, quiet, record_moments, record_central_moments, record_cumulants, record_all, osc_period, which_selection, selection_parameter, later_s, folder_label):

    
    cprint("\n\n\n\n\n\t-= neqPopDynx =-\t",'green', attrs=['underline'])
    cprint("\tNon-equilibrium\t\t\n\t  Population\t\t",'green')
    cprint("\t   Dynamics\t\t\n",'green', attrs=["underline"])
    cprint("simulator is now running...",'green')
    ###---------------------------------------------------------
    #    Settings and variable definitions
    ###---------------------------------------------------------
    initpopsize = 2*initpopsize
    BNpopsize = 2*BNpopsize
    if shom == -999:
        shom = shet*2
    if (which_selection=='step' or which_selection=='flip') and selection_parameter!=-999:
        selection_parameter=int(selection_parameter)

    if demography not in ["equilib", "dynamic", "exponential", "bottleneck", "oscillating", "tennessen", "supertennessen"]:
        cprint("\n\n\n\tERROR!! Specified demography has not (yet) been included!\n\n",'red',attrs=['underline'])
        cprint("You specified the demography:"+demography+".\n",'red',attrs=['underline'])
        cprint("Please choose from:\n\n\t'equilib', 'dynamic','exponential','bottleneck',\n\t'oscillating', 'tennessen', or 'supertennessen'\n\nor email dbalick@gmail.com with requests.\n\n\n\nExiting...\n\n\n\n", 'red')
        sys.exit()

        
    ###---------------------------------------------------------
    #   Output NON-DEMOGRAPHIC parameters to terminal
    ###---------------------------------------------------------
    cprint("\nDemographic parameters:",attrs=['underline'])
    print("demography =",demography)
    if burnin:
        print("burn-in generations =",burnin)
    if demography not in'tennessen' and demography!='supertennessen':
        print("demography generations =",generations)
        print("2N(t=0) =",initpopsize)
        if demography=='exponential':
            if int(initpopsize*np.exp(growth_rate*generations))>1e9:
                cprint("\nWARNING: This may crash due to excessive exponential growth!\n(Use lower growth/fewer generations to avoid this)\n",'red')
            print("\ngrowth rate =",growth_rate)
        elif demography=='exponential' and growthrate<=0:
            cprint("\nWARNING: growth rate is negative or zero\n", 'red')
            print("\ngrowth rate =",growth_rate)
        elif demography=='bottleneck':
            if BNpopsize>0 and BNduration>=0:
                print("2N_bottleneck =", BNpopsize)
                print("BN start generation =", BNstart)
                print("BN duration =", BNduration)
            else:
                cprint("\n\n\n\tERROR!! BNsize and/or BNduration <= 0\n\n",'red',attrs=['underline'])
                cprint("Exiting...\n\n\n\n",'red')
                sys.exit()
        elif demography=='oscillating':
            if BNpopsize>0 and osc_period>0:
                print("2N_max =", initpopsize)
                print("2N_min =", BNpopsize)
                print("oscillation period (time of one cycle) =", BNstart)
            else:
                cprint("\n\n\n\tERROR!! Nmin and/or oscillation period <= 0\n\n",'red',attrs=['underline'])
                cprint("Exiting...\n\n\n\n",'red')
                sys.exit()
        elif demography!='equilib':
            cprint("ERROR: SOMETHING WENT WRONG!!\n",'red',attrs=['underline'])
            cprint("I don't recognize the demography:\t'"+demography+"'.\n",'red')
            cprint("\n\n\nExiting...\n\n\n",'red')
            sys.exit()
    else:
        if demography=='tennessen':
            print("ancestry =",ancestry)
        else:
            print("ancestry = european (set by demography)")
        print("demography generations = 5720 (set by demography)")
        print("2N(t=0) = 28948 (set by demography)")
        generations = 5716
        initpopsize = 28948  ### THIS IS THE DEFAULT ANCESTRAL SIZE (tennessen/supertennessen)
        if demography=='supertennessen':
            print("\ngrowth rate =",growth_rate,"(second exponential epoch)")

    
    ###---------------------------------------------------------
    #   Output NON-DEMOGRAPHIC parameters to terminal
    ###---------------------------------------------------------
    cprint("\nNon-demographic parameters:",attrs=['underline'])
    print("L =",L)
    print("mu =", mu)
    print("mub =", mub)
    print("s_hom =",shom)
    print("s_het =",shet)
    print("initial condition =", initcond)
    if initcond=='delta':
        print("p(t=0) =", initcounts/initpopsize)

    if which_selection not in ['flip', 'step', 'exp', 'constant']:
        cprint("\n\nERROR: SOMETHING WENT WRONG!!\n",'red',attrs=['underline'])
        cprint("This type of time-dependent selection is not implemented:\t'"+which_selection+"'.\n",'red')
        cprint("Please choose from: 'constant', 'flip', 'step', or 'exp'.\n\n\n",'red')
        cprint("\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    elif which_selection=='flip' and selection_parameter==-999:
        cprint("\n\nERROR: TIME DEPENDENT SELECTION IS MISSPECIFIED!!!\n",'red',attrs=['underline'])
        cprint("For --skernel 'flip' must specify time of selection's sign reversal --stime <int> or --sparameter <int>.\n",'red')
        cprint("\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    elif which_selection=='step' and (selection_parameter==-999 or later_s==-999):
        cprint("\n\nERROR: TIME DEPENDENT SELECTION IS MISSPECIFIED!!!\n",'red',attrs=['underline'])
        cprint("For --skernel 'flip' must specify time of change in selection --newStime <int> or --sparameter <int> ,\n and the new selection coefficient after that time --Snew <float>.\n",'red')
        cprint("\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    elif which_selection=='exp' and selection_parameter==-999:
        cprint("\n\nERROR: TIME DEPENDENT SELECTION IS MISSPECIFIED!!!\n",'red',attrs=['underline'])
        cprint("For --skernel 'exp' must specify exponential growth rate (for selection) --sgrowth <int> or --sparameter <int>.\n",'red')
        cprint("\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    elif (which_selection in ['exp', 'flip']) and later_s!=-999:
        cprint("\n\nERROR: TIME DEPENDENT SELECTION IS MISSPECIFIED!!!\n",'red',attrs=['underline'])
        cprint("For skernel != 'step' the variable --Snew <float> cannot be specified.\n",'red')
        cprint("\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    if which_selection!='constant':
        cprint("\nTime-dependent selection parameters:",attrs=['underline'])
        print("selection kernel=", which_selection)
        if which_selection=='exp' and selection_parameter!=-999:
            print("selection growth rate=", selection_parameter)
        elif which_selection=='step' and selection_parameter!=-999:
            print("time of shet change=", selection_parameter)
            print("shet(after change)=", later_s)
        elif which_selection=='flip' and selection_parameter!=-999:
            print("sign flips at time=", selection_parameter)

    if run_number>0:
        print("[ run ",run_number,"]\n")
    else:
        print("[ time labeled run ]\n")

    ###---------------------------------------------------------
    #   Override options for moments to print to file
    ###---------------------------------------------------------
    if record_all:
        record_moments=True
        record_central_moments=True
        record_cumulants=True
    elif not record_moments and not record_central_moments and not record_cumulants:
        record_moments=True
    
    ###---------------------------------------------------------
    # PRINT POPULATION SUMMARY STATS EVERY print_gen GENERATIONS
    ###---------------------------------------------------------
    ##  Note: this is automatically turned off during burnin
    print_gen = int(printgen)
    if logprint:
        maxpg = int(np.log10(generations))
        if generations>=100:
            logprint_gen=int(10**np.linspace(0,maxpg, num=100))
            logprint_gen=np.unique(logprint_gen)
        print_gen=-999
    else:
        logprint_gen=logprint


    
    #  initialize population with desired delta function initial frequency
    if initcond == 'delta' and initcounts<initpopsize:
        initC = initcounts
        counts = initC*np.ones(L)
    elif initcond == 'delta' and initcounts >= initpopsize:
        cprint("\n\n\nWarning: initcounts is greater than initpopsize!!\nOverriding command-line input and setting to max initfreq = (initpopsize - 1)/initpopsize",'red')
        initC = (initpopsize - 1)
        counts = initC*np.ones(L)
    elif initcond == 'random':
        counts = np.random.randomint(initpopsize, size=L)
    else:
        cprint("\n\n\n\tERROR!! Choice of initial conditions not supported (yet)!\n\n",'red',attrs=['underline'])
        cprint("Please choose from 'delta' or 'random',\n\tor email dbalick@gmail.com for requests.\n\n\nExiting...\n\n\n",'red')
        sys.exit()
    counts=counts
    fixations = np.zeros(L)



    ###---------------------------------------------------------
    ##                  START BURNIN
    ###---------------------------------------------------------
    burnin_printgen=-999
    burnin_printgen=-999

    if burnin>0:
        burn_start = time.time()
        if not quiet: cprint("Burn-in starting at \u231B "+str(round((burn_start-start_time)/60,2))+" minutes \u231B",'yellow')
        maxgen = burnin
        new_popsize=initpopsize
        for gen in range(burnin):
            if burnin>10 and gen%int(burnin/10)==0:
                if gen==0:
                    if not quiet: print("\n")
                if not quiet: print("Burnin generation =",gen)
                cprint("\u231B "+str(round((time.time()-start_time)/60,2))+" minutes \u231B\n",'yellow')
            current_popsize = new_popsize
            counts, fixations = evolve(counts, shet, shom, L, mu, mub, current_popsize, gen, maxgen, fixations, 'constant', selection_parameter, later_s, userseed)
            new_popsize = demography_resize('equilib', gen, current_popsize, initpopsize, -999, -999, -999, -999, ancestry, -999)
        burn_finish=time.time()
        if not quiet: cprint("\n\nBurn-in completed at \u231B "+str(round((burn_start-start_time)/60,2))+" minutes \u231B\n\n",'yellow')
    else:
        new_popsize =initpopsize


    ###---------------------------------------------------------
    ##              START DEMOGRPAHY
    ###---------------------------------------------------------
    if not skipdemog:
        maxgen = generations
        #### clear fixations
        fixations = np.zeros(L)
        mean_freq = np.array([])
        homozygosity = np.array([])
        moment3 = np.array([])
        moment4 = np.array([])
        moment5 = np.array([])
        var_freq = np.array([])
        skew = np.array([])
        kurt = np.array([])
        central_moment5 = np.array([])
        cumulant1 = np.array([])
        cumulant2 = np.array([])
        cumulant3 = np.array([])
        cumulant4 = np.array([])
        cumulant5 = np.array([])
        fixed_sites = np.array([])
        record_gen = np.array([])
        tennessen_second_epoch=initpopsize
        demogstart_time = time.time()
        old_popsize = new_popsize
        if not quiet: cprint("Demography started at \u231B "+str(round((demogstart_time-start_time)/60,2))+" minutes \u231B",'yellow')
        
        if demography!="exponential" and demography!="supertennessen":
            growth_rate = -999
        for gen in range(generations):
            if demography=="bottleneck" and gen==BNstart:
                if not quiet: print("\n\n\nBottleneck started at generation",gen,"\n\n")
            if (gen>10 and gen%int(generations/10)==0) and not quiet:
                if gen==0:
                    print("\n\n")
                print("Generation =",gen)
                cprint("\u231B "+str(round((time.time()-start_time)/60,2))+" minutes \u231B  ("+str(int(100*gen/generations))+"% done)\n",'yellow')
            ## Set new population size, but keep old size
            past_popsize = old_popsize
            current_popsize = new_popsize
            ## Record moments of the freq. distribution as specified
            if (gen==0 or gen%print_gen==0 or gen<print_gen or gen==(generations-1)) and not logprint_gen:
                mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen=compute_stats(counts, current_popsize, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, fixations, gen, record_moments, record_central_moments, record_cumulants)
            elif demography=='bottleneck' and gen>=BNstart and gen<=(BNstart+BNduration):
                mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen = compute_stats(counts, current_popsize, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, fixations, gen, record_moments, record_central_moments, record_cumulants)
            ## Record on a log time scale (for plotting in log space)
            elif logprint_gen and (gen in logprint_gen or gen==0 or gen<logprint_gen[0] or gen==(generations-1)):
                mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen = compute_stats(counts, current_popsize, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, fixations, gen, record_moments, record_central_moments, record_cumulants)
            ## Dynamically determine tennessen initial size for second exp epoch
            if gen==5716:
                tennessen_second_epoch=current_popsize
            ## Resize population according to demography
            old_popsize = current_popsize
            new_popsize = demography_resize(demography, gen, current_popsize, initpopsize, growth_rate, BNstart, BNduration, BNpopsize, ancestry, tennessen_second_epoch, osc_period)
            ## Evolve population for one generation: mutation, selection, drift
            counts, fixations = evolve(counts, shet, shom, L, mu, mub, new_popsize, old_popsize, gen, maxgen, fixations, which_selection, selection_parameter, later_s, userseed)
        demogfinished_time = time.time()
        if not quiet: cprint("\nDemography completed at \u231B "+str(round((demogfinished_time-start_time)/60,2))+" minutes \u231B",'yellow')

    if not quiet: print("\nPrinting to file...\n")
    print_moments(shet, shom, mu, mub, L, initpopsize, demography, ancestry, growth_rate, BNpopsize, BNstart, BNduration, osc_period, burnin, generations, initcond, initcounts, printgen, run_number, mean_freq, homozygosity, moment3, moment4, moment5, var_freq, skew, kurt, central_moment5, cumulant1, cumulant2, cumulant3, cumulant4, cumulant5, fixed_sites, record_gen, current_popsize, record_moments, record_central_moments, record_cumulants, which_selection, selection_parameter, later_s, folder_label)

    runfinished_time = time.time()
    if not quiet: cprint("Total runtime: \u231B "+str(round((runfinished_time-start_time)/60,2))+"minutes \u231B\n\n",'yellow')
    cprint("\n  -= neqPopDynx run finished =-  \n",'green', attrs=['underline'])
    cprint("    Thanks for using nPD \U0001f600\n\n\n\n",'blue')




#______________________________________________________________
#______________________________________________________________
#
#                   COMMAND LINE INPUTS
#______________________________________________________________
#______________________________________________________________



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--seed", type="int", dest="userseed", default=0, help="Specify seed number for numpy.random for reproducibility; NOTE: do not use seed=0 since this indicates no seed; default 0")
    parser.add_option("--shet", type="float", dest="shet", default=-0.01, help="heterozygote selection coefficient; default -0.01")
    parser.add_option("--shom", type="float", dest="shom", default=-999, help="homozygote selection coefficient; default -999 (if not defined, 2*shet)")
    parser.add_option("-L", "--L","--length","--sites", "--numsites", type="int", dest="L", default=10000, help="Number of independent simulated sites; default 10000")
    parser.add_option("-M", "--murate", "--mutationrate", "--mu", type="float", dest="mu", default=1e-8, help="mutation rate; default 1e-6")
    parser.add_option("--mub", "--mb", "--backmutationrate", "--muback", type="float", dest="mub", default=1e-10, help="back mutation rate; default 1e-8 (mub=0 is off)")
    parser.add_option("-D", "-d","--demog", type="str", dest="demography", default="equilib", help="specify demographic model from the following options (most not yet working): 'equilib','dynamic', 'exponential','bottleneck','oscillating'/'osc'/'oscillate','tennessen', 'supertennessen' ; default 'equilib'")
    parser.add_option("--tennessenancestry","--ancestry", type="str", dest="ancestry", default="European", help="Specify ancestry for Tennessen/SuperTennessen demography from: 'european' or 'african' ; default 'european'")
    parser.add_option("--grate", "--growth", "--growthrate", type="float", dest="growth_rate", default=0.003, help="(exponential) growth rate; only used if demog='exponential' (exponential growth) or 'supertennessen' (second exponential epoch); default 0.003 (rate inferred from ExAC NFE for SuperTennessen)")
    parser.add_option("--N0", "--initN", "--initpopsize", type="int", dest="initpopsize", default=10000, help="Initial diploid (half haploid) population size at t=0; ; default 10000")
    parser.add_option("--startBN","--T0", "--t0", "--tzero", "--bottleneckstart", "--BNstarttime", "--BNstart", "--bnstart", type="int", dest="BNstart", default=1000, help="If demog=='bottleneck', first generation of the bottleneck ; default 1000")
    parser.add_option("--TBN","--TB","--tB", "--Tb","--tb", type="int", dest="BNduration", default=1000, help="If demog=='bottleneck', bottleneck duration (occurs from start to start+duration) ; default 1000")
    parser.add_option("--nb", "--Nb", "--nB", "--NB","--NBN","--BNpopsize","--BNsize","--Nmin", type="int", dest="BNpopsize", default=0, help="If demog=='bottleneck', (diploid) bottleneck size =N_BN (will run as 2N_BN); If demog=='oscillating', smallest (diploid) size during the cycle (can specify as '--Nmin'); default 1000")
    parser.add_option("--period","--oscillationtime","--osctime", type="int", dest="osc_period", default=0, help="If demog=='oscillating', this is the time for one fill cycle. ; default 0 (not using oscillating bottleneck)")
    parser.add_option("--gen","--generations","-t","--time","--runtime", type="int", dest="generations", default=10000, help="Runtime in number of generations; use 0 to only use burnin (equilib demog) and not print any temporal data; default 10000")
    parser.add_option("--burnin", type="int", dest="burnin", default=0, help="Number of generations for Burn-in (use 10*initpopsize to fully equilibrate before demography); default 0 (if not specified burnin=0 for transient dynamics)")
    parser.add_option("--printrate","--printgen","--printtime", type="int", dest="printgen", default=100, help="Ouput data (e.g., moments) every XXX generations; default 10")
    parser.add_option("--loggen", action="store_true", default=False, help="Output data (e.g., moments) in log10 spaced time intervals (e.g., gen 10, 100, 1000, 10000); default FALSE")
    parser.add_option("--skipdemog", action="store_true", default=False, help="Skip demography, only report results for burn-in; USE FOR DEBUGGING ONLY; default FALSE")
    parser.add_option("--run", type="int", dest="run_number", default=0, help="Specify run number for file labels (only if not using seed); NOTE: do not use run=0 since this indicates no label; default 0")
    parser.add_option("--initcond", "--initialcondition", "--initcondition", "--initialize", type="str", dest="initcond", default="delta", help="Specify initial condition for first generation.  Choose from: 'delta', 'random', 'initdist'(not yet working); default='delta'(freq=np.ones(L))")
    parser.add_option("--initcount", "--initcounts", type="int", dest="initcounts", default=1, help="Set initial condition to count = XXX (must be <2*initpopsize); ONLY active if initcondition='delta'(default); default=1 (=1/2N)")
    parser.add_option("-Q", "--quiet", "--shutup", dest="quiet", action="store_true", default=False, help="Quiet mode; activate to report less output (e.g.,generation and time output); default=False")
    parser.add_option("--moments", "--recordmoments", "--printmoments", "--record-moments", dest="record_moments", action="store_true", default=False, help="Print non-central moments of the frequency distribution to file (i.e., E[p^n] ); default=False (But activated if not printing central moments and/or cumulants)")
    parser.add_option("--centralmoments", "--recordcentral", "--central", "--printcentral", "--record-central-moments", dest="record_central_moments", action="store_true", default=False, help="Print central moments of the frequency distribution to file (i.e., E[(p-E[p])^n] ); default=False")
    parser.add_option("--cumulants", "--recordcumulants", "--printcumulants", "--record-cumulants", dest="record_cumulants", action="store_true", default=False, help="Print cumulants of the frequency distribution to file (i.e., K_n ); default=False")
    parser.add_option("--recordall", "--printall", "--PALL", dest="record_all", action="store_true", default=False, help="Print non-central moments, central moments, and cumulants of the frequency distribution to file (Overrides --moments, --central, and --cumulants); default=False")
    parser.add_option("--stype", "--Stype", "--skernel", "--Skernel", "--SofT", type="str", dest="which_selection", default="constant", help="Choose selection kernel from:'constant'(same shet throughout time);'step'(step function with selection shet=$later_s at time t=$TnewS);'exp'(exponential in time with growth rate $sgrow and initial selection shet=$shet);'flip'(sign of selection reverses s -> -s); Note: selection is unchanged during burnin; Note: selection is additive unless $skernel='constant'; default 'constant'(no time dependence)")
    parser.add_option("--sparam","--sparameter","--sgrow","--sgrowth","--Sgrow","--Sgrowth","--Srate","--newStime","--newstime","--stime","--Stime", type="float", dest="selection_parameter", default=-999, help="If $skernal='exp', this is the exponential growth rate; If $skernel='step', this is the time (rounded to integer) when selection changes to shet=$snew; If $skernel='flip', this is the time when selection reverses sign (shet -> -shet); default -999(not used)")
    parser.add_option("--snew","--slater","--laters","--laterS","--Snew","--Slater","--newS","--news", type="float", dest="later_s", default=-999, help="If using $skernel='step', this is the new selection coefficient shet (and shom=2*shet); default -999(not used)")
    parser.add_option("--label","--folder","--folderlabel","--labelfolder","--foldername","--namefolder", type="str", dest="folder_label", default="-999", help="Put all generated files for this run in a folder with prefix $label; e.g.,'--label test_runs'; default='-999' (no label)")
    opts, args = parser.parse_args()
    

    main(opts.userseed, opts.shet, opts.shom, opts.L, opts.mu, opts.mub, opts.demography, opts.ancestry, opts.growth_rate, opts.initpopsize, opts.BNstart, opts.BNduration, opts.BNpopsize, opts.generations, opts.burnin, opts.printgen, opts.loggen, opts.skipdemog, opts.run_number, opts.initcond, opts.initcounts, opts.quiet, opts.record_moments, opts.record_central_moments, opts.record_cumulants, opts.record_all, opts.osc_period, opts.which_selection, opts.selection_parameter, opts.later_s, opts.folder_label)

