#############################################################
# FlexDxTB model.
# Based on a TB diagnostic selection model
# David Dowdy, Jason Andrews, Peter Dodd, Robert Gilman
# rewritten version, wrapped in GUI
#############################################################


############################################################
# 2. IMPORT PACKAGES
############################################################
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.optimize import brentq # todo
import csv
from itertools import *
# for gui
from Tkinter import *
from tkMessageBox import askokcancel
from tkFileDialog import asksaveasfilename, askopenfilename
import sys, os

###############################################################
# DEFINE PARAMETERS
###############################################################

# def readparz(filename):
#     print 'Reading data from ' + filename + '...'
#     store = np.genfromtxt(filename,delimiter=',',skip_header=1,dtype=['a32','a32','a32'],filling_values='-0') # a numpy array
#     for i in range(len(store)):
#         store[i][0] = store[i][0].strip()
#         store[i][1] = store[i][1].strip()
#         store[i][2] = store[i][2].lstrip()
#         print store[i]
#         exec 'global ' + store[i][0] # declare as global
#         exec store[i][0]+'='+store[i][1] in globals() # assign value




#------------  parameters
# natural history parameters
mubl = 1.0/55                   # bl mortality
muh = 0.05                      # hiv mortality
mut = np.array([.07,.23])       # tb mortality by i
muth = 1.0                      # hiv/tb mortality
nu = np.array([.1,.27])         # spontaneous cure, by i status
pi = np.array([.14,.47])        # primary progression
v = .79                         # LTBI protection
eps = np.array([.0005,.05])     # endogenous activation
zeta = 0.22                     # reduction in infectiousness
psi = np.array([.63,.5])        # proportion TB smr+/- by h
tau0 = .01                      # rate dx in TB- 

deltae = np.array([12.0/9,12.0])   # progression rates HIV+/-
deltaf = 2.0                    # rate of finishing (wrong) treatment
# parameters that are to be fitted
theta = .01                     # hiv incidence
beta = 10                       # infectiousness parameter
phim = .707                     # MDR fitness
phi = np.array([1.0,1-.25*(1-phim),phim]) # relative fitness strains - now defined in flow

# dx scenario parameters (primary information)
dxrate = 5
ltfu = 0.15              # 15% initial default proportion, smear/GXP
emp_tx = 0.25            # 25% of pts treated empirically
ltfu_cx = 0.25           # 25% initial default proportion, culture

# various tx fail probabilities
fail_s = 0.04            # probability of failure or relapse, DS-TB
fail_rt = fail_s         # probability of failure or relapse, retreatment DS-TB
fail_i1 = 0.21           # prob of failure or relapse, INHr treated with 1st-line
fail_i2 = 0.16           # prob of failure or relapse, INHr treated with streptomycin
fail_m1 = 0.50           # prob of failure or relapse, MDR-TB treated with 1st-line
fail_m2 = 28./93.        # prob of failure or relapse, MDR-TB treated w/ 2nd-line
# turned into success probs:
p1suc = np.zeros((2,2,3,2))               # probabilities of 1st line tx success
p1suc[:,0,0,:] = 1-fail_s; p1suc[:,1,0,:] = 1-fail_rt;   # DS
p1suc[:,0,1,:] = 1-fail_i1; p1suc[:,1,1,:] = 1-fail_i2; p1suc[:,:,2,:] = 1-fail_m1;    # DR
p2suc = np.zeros((2,2,3,2))               # probabilities of 2nd line tx success
p2suc[:,0,0,:] = 1-fail_s; p2suc[:,1,0,:] = 1-fail_rt   # DS
p2suc[:,:,1,:] = 1-fail_i2; p2suc[:,:,2,:] = 1-fail_m2   # DR
# resistance acquisition
acq_s = 1*0.003/fail_s     # proportion of failed treatments acquiring resistance, DS-TB
acq_mdr = 1*0.33           # proportion of acq resi among DS-TB that is MDR (vs. INH-mono)
acq_i1 = 1*0.045/fail_i1   # proportion of failed treatments acquiring resistance, INHr tx w/ 1st-line
acq_i2 = 1*0.017/fail_i2   # proportion of failed treatments acquiring resistance, INHr tx w/ 2nd-line

# test characteristics
cx_sens = 0.85           # Sensitivity of single culture
gxp_sens = 0.72          # Sensitivity of single GXP
sm_spec = 0.98           # specificity of smear
cx_spec = 0.98           # specificity of culture
gxp_spec = 0.98          # specificity of GXP
cxr_sens = 0.98          # sensitivity of MODS for RIF resistance
cxi_sens = cxr_sens      # sensitivity of MODS for INH resistance
cxr_spec = 0.994         # specificity of MODS for RIF
cxi_spec = 0.958         # specificity of MODS for INH
gxr_sens = 0.944         # sensitivity of GXP for RIF
gxr_spec = 0.983         # specificity of GXP for RIF

# tx delays
t_smgxp = 7.0/365        # delay from smear/GXP result to tx: 7 days
t_cx = 30.0/365          # delay from culture result to tx: 30 days
t_cxr = 30.0/365         # time to culture-based DST is the same (30 days)
Te = t_smgxp             # empirical tx delay 

# costs
sm_cost = 2.             # cost of smear ($2)
gxp_cost = 15.           # cost of GXP ($15)
cx_cost =  20.           # cost of MGIT ($20)
mods_cost = 5.           # cost of MODS ($5)
dst_cost = 20.           # additional cost of DST for MGIT ($20)
sd_cost = sm_cost        # Additional cost of same-day smear
sdgxp_cost = gxp_cost    # Additional cost of same-day GXP
drug1_cost = 500
drug2_cost = 2.*drug1_cost  # Assume that cat2 costs twice as much as cat1
drug3_cost = 10.*drug1_cost # Assume that 2nd-line/MDR costs 10 times as much
outpt_cost = drug1_cost/50. # Outpatient visit costs 2% of a full course of therapy
c1 = np.zeros((2,2,3,2))                  # cost of 1st line tx
c1[:,0,:,:] = drug1_cost; c1[:,1,:,:] = drug2_cost
c2 = drug3_cost * np.ones((2,2,3,2))                  # cost of 2nd line tx

# prop inappproprate tx that def vs fail (I->A vs I->P)
defvfail_s = 6./7.       # proportion of inappropriate tx's that are default (vs. failure), DS-TB
defvfail_i = 2./3.       # proportion of inapp tx's that are default, INHr
defvfail_m = 11./25.     # proportion of inapp tx's that are default, MDR
omega = defvfail_s * np.ones((2,2,3,2))
omega[:,:,1,:] = defvfail_i; omega[:,:,2,:] = defvfail_m
pdefcure = .5                            # proportion defaulters (from I) cured

# for holding the above
ndxtx = np.zeros((2,2))           # dx & tx costs for non-TB

# global array parameters set in below function 'setparameters'
SNtb = np.zeros((2,2,3,2))                # sensitivity of test for TB
SNdr = np.zeros((2,2,3,2))                # sensitivity of test for DR
SPdr = np.zeros((2,2,3,2))                # specificity of test for DR
Ttb = np.zeros((2,2,3,2))                 # delay to 1st line tx
Tdr = np.zeros((2,2,3,2))                 # delay to 2nd line tx
LTFU = np.zeros((2,2,3,2))                # loss to follow up 
cdx = np.zeros((2,2,3,2))                 # cost of dx
spec = np.zeros(2)                # test specificity - for misdx
DR = np.zeros((2,2,3,2))              # identified as DR


###########################################################
# RELEVANT FUNCTIONS
###########################################################


# some diagnostic scenarios
# the quantities normally changed in each scenario are:
# 
# SNtb  = the sensitivity of the algorithm for detecting TB
# SNdr  = the sensitivity of the algorithm for detecting MDR, conditional on detection
# SPdr  = the specificity of the algorithm for detecting MDR, conditional on detection
# Ttb   = the delay to first line tx
# Tdr   = the delay to second line tx
# LTFU  = the loss to follow up
# cdx   = the cost of a diagnostic attempt
# spec  = the specificity of the algorithm for TB in those w/o TB
# ndxtx = the cost of diagnosing and treating someone who does not have TB in this algorithm



def setparameters( scenario ):
    print 'calculating parameters for scenario', scenario, '...'
    unknown = False                                        # tracking if option known
    p1sucL = p1suc.copy()                                  # may be different
    for k in product(range(2),range(2),range(3),range(2)):   # loop over hpdi
        h,p,d,i = k                                          # hpdi=(HIV, previous tx, dr, smr)
        if scenario == 0:                                    # "0 = baseline" 
            SNtb[h,p,d,i] = 0 if i==0 else 1    # smear
            SNdr[h,p,d,i] = cxr_sens if (p==1 and i==1 and d==2) else 0
            SPdr[h,p,d,i] = cxr_spec if (p==1 and i==1 and d!=2) else 1
            Ttb[h,p,d,i] = t_smgxp        # 1st line delay			
            Tdr[h,p,d,i] = t_cxr                   # 2nd line delay
            LTFU[h,p,d,i] = ltfu#ltfu_cx if i==1 and p==1 else ltfu
            cdx[h,p,d,i] = (sm_cost + cx_cost + outpt_cost + dst_cost) if p==1 and i==1 else (sm_cost + outpt_cost)
            spec[h] = sm_spec	
            ndxtx[h,p] = sm_cost + outpt_cost + ( (1-sm_spec)*drug1_cost  if p==0  else ( (1-sm_spec)*(drug2_cost + cx_cost) + (1-sm_spec)*(1-cx_spec)*dst_cost + (1-sm_spec)*(1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)) )

        elif scenario == 1:               # "1 = culture if previously treated"
            SNtb[h,p,d,i] = (cx_sens if i==0 else 1) if p==1 else (0 if i==0 else 1)
            SNdr[h,p,d,i] = cxr_sens if p==1 and d==2 else 0
            SPdr[h,p,d,i] = cxr_spec if p==1 and d!=2 else 1
            Ttb[h,p,d,i] = t_cx if p==1 and i==0 else t_smgxp
            Tdr[h,p,d,i] = t_cxr                   # 2nd line delay
            LTFU[h,p,d,i] = ltfu_cx if p==1 and i==0 else ltfu 
            cdx[h,p,d,i] = (sm_cost + cx_cost + outpt_cost + dst_cost) if p==1 else (sm_cost + outpt_cost) 
            spec[h] = sm_spec
            ndxtx[h,p] = sm_cost + outpt_cost + ( (1-sm_spec)*drug1_cost  if p==0 else ( cx_cost + (1-cx_spec)*dst_cost + (1-sm_spec)*drug2_cost + (1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost) ) )

        elif scenario == 2:   # "2 = Xpert for all"
            SNtb[h,p,d,i] = gxp_sens if i==0 else 1
            SNdr[h,p,d,i] = gxr_sens if d==2 else 0
            SPdr[h,p,d,i] = gxr_spec if d!=2 else 1
            Ttb[h,p,d,i] = t_smgxp
            Tdr[h,p,d,i] = t_smgxp           
            LTFU[h,p,d,i] = ltfu
            cdx[h,p,d,i] =  (gxp_cost + outpt_cost) 
            spec[h] = gxp_spec
            acost = drug1_cost if p==0 else drug2_cost
            ndxtx[h,p] = gxp_cost + outpt_cost + (1-gxp_spec)*acost + (1-gxp_spec)*(1-gxr_spec)*(drug3_cost-acost)

        elif scenario == 3:        # "3 = MODS/TLA"
            # N.B. p1suc  is different in MODS/TLA - see bottom portion of function for code
            SNtb[h,p,d,i] = cx_sens if i==0 else 1
            SNdr[h,p,d,i] = cxr_sens if d==2 else 0
            SPdr[h,p,d,i] = cxr_spec if d!=2 else 1
            Ttb[h,p,d,i] = t_smgxp if i==1 else t_cx
            Tdr[h,p,d,i] = t_cxr           
            LTFU[h,p,d,i] = ltfu if i==1 else ltfu_cx
            cdx[h,p,d,i] =  (sm_cost + mods_cost + outpt_cost) 
            spec[h] = cx_spec
            acost = drug1_cost if p==0 else drug2_cost
            ndxtx[h,p] = sm_cost + outpt_cost + mods_cost + (1-cx_spec)*acost + (1-cx_spec)*(1-cxr_spec)*(drug3_cost-acost)
            
        elif scenario == 4:  # "4 = same-day smear"
            SNtb[h,p,d,i] = 0 if i==0 else 1    # smear
            SNdr[h,p,d,i] = cxr_sens if i==1 and p==1 and d == 2 else 0
            SPdr[h,p,d,i] = cxr_spec if i==1 and p==1 and d != 2 else 1
            Ttb[h,p,d,i] = 1.0 / 365 
            Tdr[h,p,d,i] = t_cxr
            LTFU[h,p,d,i] = 0
            cdx[h,p,d,i] =  (sm_cost + cx_cost + outpt_cost + dst_cost + sd_cost) if i==1 and p==1 else (sm_cost + outpt_cost + sd_cost)
            spec[h] = sm_spec   # nonTB test spec
            ndxtx[h,p] = sm_cost + outpt_cost + sd_cost + ( (1-sm_spec)*drug1_cost  if p==0  else ( (1-sm_spec)*(drug2_cost + cx_cost) + (1-sm_spec)*(1-cx_spec)*dst_cost + (1-sm_spec)*(1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)) )
                   
            
        elif scenario == 5:  # "5 = same-day Xpert"
            SNtb[h,p,d,i] = gxp_sens if i==0 else 1
            SNdr[h,p,d,i] = gxr_sens if d==2 else 0
            SPdr[h,p,d,i] = gxr_spec if d!=2 else 1
            Ttb[h,p,d,i] = 1.0 / 365
            Tdr[h,p,d,i] = 1.0 / 365
            LTFU[h,p,d,i] = 0
            cdx[h,p,d,i] =  gxp_cost + outpt_cost + sdgxp_cost
            spec[h] = gxp_spec
            acost = drug1_cost if p==0 else drug2_cost
            ndxtx[h,p] = gxp_cost + outpt_cost + sdgxp_cost + (1-gxp_spec)*acost + (1-gxp_spec)*(1-gxr_spec)*(drug3_cost-acost)

            
        elif scenario == 6:      #"6 = Xpert for smear-positive only"         
            SNtb[h,p,d,i] =  0 if i==0 else 1    # smear
            SNdr[h,p,d,i] = gxr_sens if d==2 and i==1 else 0
            SPdr[h,p,d,i] = gxr_spec if d!=2 and i==1 else 1
            Ttb[h,p,d,i] = t_smgxp
            Tdr[h,p,d,i] = t_smgxp
            LTFU[h,p,d,i] = ltfu 
            cdx[h,p,d,i] = (gxp_cost + outpt_cost + sm_cost) if i==1 else (outpt_cost + sm_cost)
            spec[h] = sm_spec   # nonTB test spec
            ndxtx[h,p] = sm_cost + outpt_cost + ( (1-sm_spec)*drug1_cost  if p==0  else ( (1-sm_spec)*(drug2_cost + cx_cost) + (1-sm_spec)*(1-cx_spec)*dst_cost + (1-sm_spec)*(1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost))  )
            
        elif scenario == 7:                  # "7 = Xpert for HIV-positive only"
            SNtb[h,p,d,i] = (gxp_sens if i==0 else 1) if h==1 else (0 if i==0 else 1)    
            SNdr[h,p,d,i] =  (gxr_sens if d==2 else 0) if h==1 else (cxr_sens if p==1 and i==1 and d==2 else 0)
            SPdr[h,p,d,i] = (gxr_spec if d!=2 else 1) if h==1 else (cxr_spec if p==1 and i==1 and d!=2 else 1)
            Ttb[h,p,d,i] =  t_smgxp        
            Tdr[h,p,d,i] = t_smgxp if h==1 else t_cxr    
            LTFU[h,p,d,i] = ltfu 
            cdx[h,p,d,i] = (gxp_cost + outpt_cost) if h==1 else ( (sm_cost + cx_cost + outpt_cost + dst_cost) if p==1 and i==1 else (sm_cost + outpt_cost) )
            spec[h] = gxp_spec if h==1 else sm_spec
            if h==0:
                ndxtx[h,p] = sm_cost + outpt_cost + ( (1-sm_spec)*drug1_cost  if p==0  else ( (1-sm_spec)*(drug2_cost + cx_cost) + (1-sm_spec)*(1-cx_spec)*dst_cost + (1-sm_spec)*(1-cx_spec)*(1-cxr_spec)*(drug3_cost-drug2_cost)) )
            else:
                acost = drug1_cost if p==0 else drug2_cost
                ndxtx[h,p] = gxp_cost + outpt_cost + (1-gxp_spec)*acost + (1-gxp_spec)*(1-gxr_spec)*(drug3_cost-acost)
                
        elif scenario == 8:                  # "8 = Xpert with culture DST confirmation"
            SNtb[h,p,d,i] = gxp_sens if i==0 else 1
            SNdr[h,p,d,i] = gxr_sens*cxr_sens if d==2 else 0 # sequential tests confirming
            SPdr[h,p,d,i] = 1-(1-gxr_spec)*(1-cxr_spec) if d!=2 else 1
            Ttb[h,p,d,i] = t_smgxp
            Tdr[h,p,d,i] = t_smgxp
            LTFU[h,p,d,i] = ltfu 
            cdx[h,p,d,i] =  gxp_cost + outpt_cost + ((1-gxr_spec) if d!=2 else gxr_sens ) * (cx_cost+dst_cost) #
            spec[h] = gxp_spec
            acost = drug1_cost if p==0 else drug2_cost
            ndxtx[h,p] =  gxp_cost + outpt_cost + (1-gxp_spec)*acost + \
                (1-gxp_spec)*(1-gxr_spec)*(cx_cost + dst_cost + (1-cxr_spec)*(drug3_cost-acost))
            
        else:
            unknown = True
    # check the option and return the parameters used on ODEs
    if unknown:        
        print "Unknown option! Using baseline..."
        return setparameters(0)        
    else:                    # txrate piece - parameters as used
        for d in range(3):      # more direct chance of going onto 2nd-line
            DR[:,:,d,:] = 1- SPdr[:,:,d,:] if d!=2 else SNdr[:,:,d,:]
        Ep = emp_tx * np.ones((2,2,3,2))              # empirical tx probability 
        p1 = (1 - LTFU)*SNtb*(1 - DR) + Ep*(LTFU + (1 - LTFU)*(1 - SNtb)) # prob 1st line tx -  needs attention in setpar's
        p2 = (1 - LTFU)*SNtb*DR
        pe = Ep*(LTFU + (1 - LTFU)*(1 - SNtb))   # empiric 1st line tx probability
        pe[p1>0] /= p1[p1>0]; pe[p1==0]=0        # conditioning on 1st line tx
        d1 = pe*Te + (1 - pe)*Ttb             # delay to 1st line
        d2 = Tdr         # total delay 2nd line
        dxr = dxrate * np.ones((2,2,3,2))#; dxr[1] *= muth/mut[0] 
        sigmaL = dxr * (p1 * p1sucL + p2 * p2suc)            # rates of dx 2 successful tx (A->P)
        kappaL = dxr * (p1 * (1-p1sucL) + p2 * (1-p2suc) )   # rates for ineffective tx - leaving A
        SHL = kappaL[:,:,0,:] * acq_s * (1-acq_mdr)         # 
        SML = kappaL[:,:,0,:] * acq_s * acq_mdr
        HML = (dxr * p1 * (1-p1sucL))[:,:,1,:] * acq_i1 + (dxr * p2 * (1-p2suc))[:,:,1,:] * acq_i2
        kappanrL = kappaL.copy()
        kappanrL[:,:,0,:] *= (1-acq_s)
        kappanrL[:,:,1,:] -= HML # the bit generating no resistance
        rhoL = 1.0 / (d1 + (d2-d1)*p2*(p2suc-p1sucL)/(p1 * p1sucL + p2 * p2suc) )   # rates of cure: post dx delays to rx (P->)
        costtxSL = (p1*p1sucL*c1 + p2*p2suc*c2)/(p1*p1sucL + p2*p2suc)   # mean cost of success
        costtxUL = (p1*(1-p1sucL)*c1 + p2*(1-p2suc)*c2)/(p1*(1-p1sucL) + p2*(1-p2suc)) # mean cost of non-success
        
        return sigmaL, rhoL, kappaL, kappanrL, SHL, SML, HML, spec, costtxSL, costtxUL, cdx, ndxtx


# sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(0)


# ------------ functions
# parser
def parseX(X):
    if len(X) != 100:
        print "Error: incorrect length Xflow argument!"
    # extract the variables in more meaningfully structured way - hpdi
    U = X[0:4].reshape(2,2)
    L = X[4:16].reshape(2,2,3)
    E = X[16:28].reshape(2,2,3)
    A = X[28:52].reshape(2,2,3,2)
    P = X[52:76].reshape(2,2,3,2)
    I = X[76:100].reshape(2,2,3,2)
    return U,L,E,A,P,I

# mortality - currently duplicates that in flow, may replace  
def mortality(X):
    U,L,E,A,P,I = parseX(X)
    N = X.sum()
    mutb = mut[0]*( E[0].sum() + (A+P)[0,:,:,0].sum() + I[0].sum() ) \
      + mut[1]*(A+P)[0,:,:,1].sum() + muth * ( E[1].sum() + (A+P+I)[1].sum())
    mutot = muh * (U[1].sum()+L[1].sum()+E[1].sum()+A[1].sum()+P[1].sum()+I[1].sum())\
      + mutb + mubl * N
    return mutot, mutb

# get FOI
def foi(X,beta,phim):
    U,L,E,A,P,I = parseX(X)
    N = X.sum()
    phi = np.array([1.0,1-.25*(1-phim),phim]) # relative fitness strains - now defined in flow    
    lam = np.zeros(3)           # foi
    for d in range(3):          # assuming all I less infectious
        lam[d] = zeta * ( E[:,:,d].sum() + A[:,:,d,0].sum() + P[:,:,d,0].sum() + I[:,:,d,:].sum() ) \
          + A[:,:,d,1].sum() + P[:,:,d,1].sum()
        lam[d] *= beta * phi[d] / N
    return lam

# flow defining odes
def Xflow(X,t, beta,theta,phim, sigma, rho, kappa, kappanr, SH, SM, HM, spec):
    # parse into more meaningfully-structured chunks
    U,L,E,A,P,I = parseX(X)
    # construct foi and mortality
    N = X.sum()
    mutot = mortality(X)[0]
    lam = foi(X,beta,phim)
    # differential equations
    # change for U - eqn 1
    dU = -(lam.sum()+mubl) * U  # TB infection and background death
    dU[0,0] += mutot            # births
    for h in range(2):
        dU[h,0] -= tau0 * (1-spec[0]) * U[h,0] # misdx TB care        
        dU[h,1] += tau0 * (1-spec[h]) * U[h,0] # misdx TB care
    dU[0] -= theta * U[0]     # hiv infection
    dU[1] += (theta * U[0] - muh * U[1]) # hiv infection and death
    # change for L - eqn 2
    dL = -mubl * L        #  background death 
    dL[0] -= theta * L[0] # HIV incidence
    dL[1] += theta * L[0] - muh * L[1]         # HIV incidence less additional hiv mortality
    for h in range(2):
        dL[h,0] -= tau0 * (1-spec[0]) * L[h,0] # misdx TB care        
        dL[h,1] += tau0 * (1-spec[h]) * L[h,0] # misdx TB care
    Lvuln = L.copy()
    Lvuln[0] *= (1-v) # L but vulnerable to reinfection; NB copy needed in python
    dL -= lam.sum() * Lvuln      # actual reinfection
    for h in range(2):
        dL[h] -= eps[h] * L[h]  # endo activation
        for d in range(3):
            dL[h,:,d] += lam[d] * (1-pi[h]) * (U[h] + Lvuln.sum(2)[h]) # primary disease
    dL[:,1,:] += (rho*P + omega*deltaf*pdefcure*I).sum(3).sum(1)    # arrival from treatment into previously treated
    dL[0] += nu[0] * E[0]                                  # spontaneous recovery
    for i in range(2):                                     # spontaneous recovery
        dL[0] += nu[i] * (A[0,:,:,i] + P[0,:,:,i] + I[0,:,:,i])
    # change for E - eqn 3
    dE =  - mubl * E     #  background death
    dE[0] += eps[0] * L[0] - (theta + mut[0]+nu[0]+deltae[0]) * E[0] # endo act - HIV incidence - death/cure/progression HIV- 
    dE[1] += eps[1] * L[1] + theta * E[0] - (muh+muth+deltae[1]) * E[1]   # endo act + HIV incidence - death/progression HIV+
    for d in range(3):
        for h in range(2):
            dE[h,:,d] += lam[d] * pi[h] * (U[h] + Lvuln.sum(2)[h]) # primary disease
    # change for A - eqn 4
    dA = -(mubl+sigma+kappa) * A # background mortality, useful and ineffective treatment rates
    for i in range(2):           # hiv-ve mortality and self-cure
        dA[0,:,:,i] -= (mut[i]+nu[i]) * A[0,:,:,i]        
    dA[0] -= theta * A[0]          # HIV incidence
    dA[1] += theta * A[0] - (muh+muth) * A[1]  # HIV incidence less hiv+ve mortality
    dA[:,1,:,:] += (omega*deltaf*(1-pdefcure)*I).sum(1)   # defaulted not cured into previously treated
    dA[:,:,0,:] -= (SH+SM)* A[:,:,0,:] # S -> DR emergence
    dA[:,:,1,:] -= HM * A[:,:,1,:]     # H -> M DR emergence
    for h in range(2):                 # progression from early
        for i in range(2):
            dA[h,:,:,i] += deltae[h] * ( psi[h] if i==1 else 1-psi[h] ) * E[h,:,:]    
    # change for P - eqn 5
    dP = sigma * A  - (mubl + rho) * P   # dx rate less background mortality & cure rates
    dP[0] -= theta * P[0]               # HIV incidence
    dP[1] += theta * P[0] - (muh+muth) * P[1] # HIV incidence less mortality, HIV+
    for i in range(2):
        dP[0,:,:,i] -= (mut[i]+nu[i]) * P[0,:,:,i] # mortality & self-cure, HIV-
    dI = kappanr * A - (mubl + deltaf) * I  # arrival on inappropriate rx less background death & tx durn
    dI[:,1,:,:] += ((1-omega)*deltaf*I).sum(1)   # arrival into previously treated from failure
    dI[0] -= theta * I[0]       # HIV incidence
    dI[1] += theta * I[0] - (muh + muth) * I[1] # HIV incidence/mortality
    for i in range(2):                                       # mortality, self-cure and completion, HIV-
        dI[0,:,:,i] -= (mut[0] + nu[i])* I[0,:,:,i] # NB zero in nu
    dI[:,:,2,:] += SM *A[:,:,0,:] + HM *A[:,:,1,:] # S,H -> M DR emergence
    dI[:,:,1,:] += SH * A[:,:,0,:]     # S -> H DR emergence
    # join together all the flows
    dX = np.concatenate((dU.reshape((4,)),dL.reshape((12,)),dE.reshape((12,)),dA.reshape((24,)),dP.reshape((24,)),dI.reshape((24,))))
    return dX

# get some relevant epi indicators
def evalepi(X,beta,theta,phim):
    U,L,E,A,P,I = parseX(X)
    Lvuln = L.copy()
    Lvuln[0] *= (1-v)
    lam = foi(X,beta,phim)
    N = X.sum()
    inc = np.zeros((2,2,3))     # hpd
    # hang on - does this include some flows which are not ->E todo?
    for h in range(2):
        inc[h] += eps[h] * L[h]
        for d in range(3):
            inc[h,:,d] += lam[d] * pi[h] * (U[h] + Lvuln.sum(2)[h]) # primary 
    inc  /= (N + 1e-15)
    incnew = inc[:,0,:].sum()
    incretx = inc[:,1,:].sum()
    incinhnew = inc[:,0,1].sum()
    incinhretx = inc[:,1,1].sum()
    incmdrnew = inc[:,0,2].sum()
    incmdrretx = inc[:,1,2].sum()
    inctbhiv = inc[1].sum()
    TBmort = mortality(X)[1] / (N + 1e-15)
    TBprev = (E.sum() + (A+P+I).sum()) / (N + 1e-15)
    HIVprev = (U[1].sum() + (L+E)[1].sum() + (A+P+I)[1].sum()) / (N + 1e-15)
    return incnew, incretx, incinhnew, incinhretx, incmdrnew, incmdrretx, inctbhiv, TBmort, TBprev, HIVprev

# extract relevant costs
def evalcost(X, sigma, kappa, costtxS, costtxU, cdx, ndxtx): 
    U,L,E,A,P,I = parseX(X)
    cost = (sigma * A * costtxS).sum() # successful tx
    cost += (kappa * A * costtxU).sum() # unsuccessful tx
    cost += (dxrate * A * cdx).sum() # dx costs
    for h in range(2):          
        for p in range(2):
            cost += tau0 * ndxtx[h,p] * ( U[h,p] + L[h,p,:].sum() ) # 
    return cost 

# get out and calculate some relevant data from X timeseries
# incnew, incretx, incinhnew, incinhretx, incmdrnew, incmdrretx, inctbhiv, TBmort, TBprev, HIVprev, cost
def extractdata(XA,beta,theta,phim, sigma, kappa, costtxS, costtxU, cdx, ndxtx):
    nr = XA.shape[0]
    incnewz=np.zeros(nr); incretxz=np.zeros(nr); incinhnewz=np.zeros(nr); incinhretxz=np.zeros(nr); incmdrnewz=np.zeros(nr); incmdrretxz=np.zeros(nr); 
    inctbhivz=np.zeros(nr); TBmortz=np.zeros(nr); TBprevz=np.zeros(nr); HIVprevz=np.zeros(nr); costz=np.zeros(nr); Nz = np.zeros(nr)
    for i in range(nr):
        Nz[i] = XA[i,:].sum()
        incnewz[i], incretxz[i], incinhnewz[i], incinhretxz[i], incmdrnewz[i], incmdrretxz[i], inctbhivz[i], TBmortz[i], TBprevz[i], HIVprevz[i] = evalepi(XA[i,:],beta,theta,phim)
        costz[i] = evalcost(XA[i,:], sigma, kappa, costtxS, costtxU, cdx, ndxtx)
    return Nz, incnewz, incretxz, incinhnewz, incinhretxz, incmdrnewz, incmdrretxz, inctbhivz, TBmortz, TBprevz, HIVprevz, costz


# ---------------------------------------- NEW fitting attempts
def initialguess( target_incL, target_hivL, target_mdrL ):
    # initial state guess
    y0 = np.ones(100)
    U0,L0,E0,A0,P0,I0 = parseX(y0)
    fac = 1.0 / (1+target_incL*10/mubl) # something like proportion susceptible?
    U0 *= fac 
    L0 *= (1-fac)
    E0 *= target_incL/12        # spread evenly, assuming incidence is roughly prevalence
    A0 *= target_incL/24
    P0 *= target_incL/24
    I0 *= target_incL/24
    U0[0] *= (1-target_hivL); U0[1] *= target_hivL # scaling by targets
    L0[0] *= (1-target_hivL); L0[1] *= target_hivL
    E0[0] *= (1-target_hivL); E0[1] *= target_hivL
    A0[0] *= (1-target_hivL); A0[1] *= target_hivL
    P0[0] *= (1-target_hivL); P0[1] *= target_hivL
    I0[0] *= (1-target_hivL); I0[1] *= target_hivL
    L0[:,:,1] *= 2*target_mdrL; L0[:,:,2] *= target_mdrL
    E0[:,:,1] *= 2*target_mdrL; E0[:,:,2] *= target_mdrL
    A0[:,:,1,:] *= 2*target_mdrL; A0[:,:,2,:] *= target_mdrL
    P0[:,:,1,:] *= 2*target_mdrL; P0[:,:,2,:] *= target_mdrL
    I0[:,:,1,:] *= 2*target_mdrL; I0[:,:,2,:] *= target_mdrL
    U0[:,1] *= .2               # 20% previously treated
    L0[:,1] *= .2
    E0[:,1] *= .2
    A0[:,1] *= .2
    P0[:,1] *= .2
    I0[:,1] *= .2
    y0 = np.concatenate((U0.reshape((4,)),L0.reshape((12,)),E0.reshape((12,)),A0.reshape((24,)),P0.reshape((24,)),I0.reshape((24,))))
    y0 /= y0[0]
    return y0
    


def getfit( target_incL, target_hivL, target_mdrL, sigma, rho, kappa, kappanr, SH, SM, HM, spec, fitub = 5 ):
    # check not a low-incidence scenario
    if target_incL >= 50:
        print 'Attempting to fit equilibrium to:'
        print '\t TB incidence per 100,000/y=',1*target_incL
        print '\t HIV prevalence (%)=',1*target_hivL    
        print '\t MDR in new TB cases (%)=',1e2*target_mdrL        
        # rescale
        target_incL /= 1e5
        target_hivL /= 1e2

        # initial guess
        y0 = initialguess(target_incL, target_hivL, target_mdrL)
        # new  square and bound
        bds = np.array([120.0, 0.1, fitub ]) # upper bounds for beta, hiv incidence, relative fitness mdr
        z1 = np.array([1e5*target_incL / (10.0), target_hivL * muh, .7]) / bds
        z1 = z1/(1-z1)
        z1 = np.sqrt(z1)
        z0 = np.concatenate(( z1, y0[1:] ))
        def aFlow( Z ):
            X = np.concatenate((np.ones(1), abs(Z[3:]) )) # 1 for the simplest U category, now squaring
            pz = Z[0:3]*Z[0:3]
            pz = bds * pz / (pz + 1.0)
            betaLL, thetaLL, phimLL = pz
            # print betaLL, thetaLL, phimLL
            targz = evalepi(X, betaLL, thetaLL, phimLL) # [0,1,4,9]: incnew, incretx, incmdrnew, hivprev
            dX = Xflow(X,0, betaLL,thetaLL,phimLL,sigma, rho, kappa, kappanr, SH, SM, HM, spec) # the flow with the correct targets
            dY = np.array([targz[0]+targz[1] - target_incL, targz[9] - target_hivL, targz[4] / targz[0] - target_mdrL])
            # print '\t',dY        
            return np.concatenate((dY,dX[1:]))  

        test = fsolve( aFlow, z0 )
        y0L = np.concatenate((np.ones(1), abs(test[3:]) ))   
        test[0:3] = test[0:3]*test[0:3]
        test[0:3] = test[0:3] / (1+test[0:3])
        if abs(test[0:3] - np.ones(3)).min() < 1e-3:
            print 'You\'ve asked me to solve for parameters very close to my allowed range.'
            print 'I\'m afraid this might well have caused some problems. Proceed with particular caution.'
        test[0:3] *= bds
        betaL, thetaL, phimL = test[0:3]
        if phimL > 1:           # unrealistic root
            if fitub > 1:       # try constraining...
                print 'Changing MDR upper bound!'
                betaL, thetaL, phimL, y0L = getfit( 1e5*target_incL, 1e2*target_hivL, target_mdrL, sigma, rho, kappa, kappanr, SH, SM, HM, spec, fitub = 1 ) # 
            else:
                print 'Numerical trouble fitting MDR prevalence! Equilibrium fit may be wrong! Try slightly different targets.'
        else:
            # end doing square and bound
            print '...equilibrium fit done!'

    else:                       # low incidence scenario
        print 'This is a low incidence scenario.'
        print 'Fitting a decline to ', target_incL,'/100kY from an equilibrium of 50/100kY 50 years ago.'
        betaL, thetaL, phimL, y0L = getfit( 50, target_hivL, target_mdrL, sigma, rho, kappa, kappanr, SH, SM, HM, spec ) # get equilibrium fit with Inc = 50/100kY
        # rescale
        target_incL /= 1e5
        target_hivL /= 1e2
        # do the fit to decline...
        tL = np.arange(0,50,.05)
        
        # error function
        def err( betaLL  ):
            betaLL = abs(betaLL)
            yL = odeint(func=Xflow,y0=y0L,t=tL,args = (betaLL,thetaL,phimL,sigma, rho, kappa, kappanr, SH, SM, HM, spec))
            targz = evalepi(yL[-1,:], betaLL, thetaL, phimL) # get incidence
            # print  targz[0]+targz[1] - target_incL
            return targz[0]+targz[1] - target_incL

        # solve
        betaL = brentq( err, 0, betaL )
        # get y0
        yL = odeint(func=Xflow,y0=y0L,t=tL,args = (betaL,thetaL,phimL,sigma, rho, kappa, kappanr, SH, SM, HM, spec))
        y0L = yL[-1,:]
        print '...declining incidence fitted!'
        # end of low incidence scenario
    

    return betaL, thetaL, phimL, y0L # beta, theta, phi, y0

###########################################################
# TESTING - normally commented out
###########################################################

# from matplotlib import pyplot as plt

# target_inc = 698.#50.#1*200.0
# target_hiv = 5.781#1*0.83 
# target_mdr = 0.006#.05*3.7/100

# work hard to reproduce an error
# while phim < 1:
#     target_inc = 500*np.random.rand() + 51
#     target_hiv = 10*np.random.rand()
#     target_mdr = np.random.rand()/10.0
#     sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(0)
#     beta, theta, phim, y0 = getfit( target_inc, target_hiv, target_mdr, sigma, rho, kappa, kappanr, SH, SM, HM, spec )
#     print phim


# sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(0)
# beta, theta, phim, y0 = getfit( target_inc, target_hiv, target_mdr, sigma, rho, kappa, kappanr, SH, SM, HM, spec )


    
# tz = np.arange(0,50,.05)
# y = odeint(func=Xflow,y0=y0,t=tz,args = (beta,theta,phim,sigma, rho, kappa, kappanr, SH, SM, HM, spec))

# Nz, incnewz, incretxz, incinhnewz, incinhretxz, incmdrnewz, incmdrretxz, inctbhivz, TBmortz, TBprevz, HIVprevz, costz = extractdata(y,beta,theta,phim, sigma, kappa, costtxS, costtxU, cdx, ndxtx)



# # plot
# plt.close('all')
# plt.plot(tz,1e5*(incretxz+incnewz) ,label='TB inc tot')
# plt.plot(tz,1e3*incmdrnewz / (incnewz),label='TB inc MDR new * 10/%')
# plt.plot(tz,1e3*HIVprevz,label='HIV prev*10/%')
# plt.xlabel('time')
# plt.legend(loc=0)
# plt.show()


# plt.close('all')
# #incnewz, incretxz, incinhnewz, incinhretxz, incmdrnewz, incmdrretxz, inctbhivz, TBmortz, TBprevz, HIVprevz
# plt.plot(tz,1e6*incmdrnewz,label='TB inc MDR new')
# plt.plot(tz,1e6*incinhnewz,label='TB inc INH new')
# plt.plot(tz,1e6*incinhretxz,label='TB inc INH retx')
# plt.plot(tz,1e6*incmdrretxz,label='TB inc MDR retx')
# plt.plot(tz,1e6*inctbhivz,label='TB inc HIV')
# plt.plot(tz,1e5*incnewz,label='TB inc new')
# plt.plot(tz,1e5*incretxz,label='TB inc retx')
# plt.xlabel('time')
# plt.legend(loc=0)
# plt.show()

###########################################################
# FUNCTION CALLED DIRECTLY BY GUI
###########################################################

def runmodel( target_inc, target_hiv, target_mdr, cost, int_select ):

    # baseline
    sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(0)

    # do fit
    beta, theta, phim, y0 = getfit( target_inc, target_hiv, target_mdr, sigma, rho, kappa, kappanr, SH, SM, HM, spec )

    # run and analyse results
    duration = 5.                        # 5 year time horizon
    timestep = 0.01                      # calculate equations every 0.01 yrs
    time_range = np.arange(0, duration+timestep, timestep) # vector for calculations
    y = odeint(func=Xflow,y0=y0,t=time_range,args = (beta,theta,phim,sigma, rho, kappa, kappanr, SH, SM, HM, spec))
    incnew, incretx, incinhnew, incinhretx, incmdrnew, incmdrretx, inctbhiv, TBmort, TBprev, HIVprev = evalepi( y[-1,:], beta, theta, phim )
    
    print " "
    print "BASELINE"
    print "Incidence, new:", np.around( 1e5*incnew, decimals=1 ), "per 100,000"
    print "Incidence, retx:", np.around( 1e5*incretx, decimals=1), "per 100,000"
    print "Incidence, total:", np.around( 1e5*(incnew + incretx), decimals=1 ), "per 100,000"
    print "Incidence, INH new:", np.around( 1e2*incinhnew/incnew, decimals=1 ), "%"
    print "Incidence, INH retx:", np.around( 1e2*incinhretx/incretx, decimals=1 ), "%"
    print "Incidence, MDR new:", np.around( 1e2*incmdrnew/incnew, decimals=2 ), "%"
    print "Incidence, MDR retx:", np.around( 1e2*incmdrretx/incretx, decimals=2 ), "%"
    print "Incidence, MDR total:", np.around( 1e5*(incmdrretx+incmdrnew) , decimals=2), "per 100,000"
    print "Incidence, TB/HIV:", np.around( 1e2*inctbhiv/(incnew+incretx), decimals=1 ), "%"
    print "TB mortality:", np.around( 1e5*TBmort, decimals=1), "per 100,000"
    print "TB duration:", np.around( TBprev/(incnew+incretx), decimals=2 ), "years"
    print "HIV prevalence:", np.around( 1e2*HIVprev, decimals=1 ), "%"
    costvec = np.array( [ evalcost(y[i,:],sigma, kappa, costtxS, costtxU, cdx, ndxtx) for i in range(y.shape[0]) ] )
    c1 = sum(costvec[0:100])/100
    c5 = sum(costvec)/100
    print "Cost by end Year 1: $", np.around( c1 )
    print "Cost by end Year 5: $", np.around( c5 )
    print " "
    # i1 = [i for i in range(len(time_range)) if time_range[i]==1.0][0] # surely better way? todo
    # i5 = [i for i in range(len(time_range)) if time_range[i]==5.0][0] 
    #evalcost(y[i1,:],sigma, kappa, costtxS, costtxU, cdx, ndxtx)
    #evalcost(y[i5,:],sigma, kappa, costtxS, costtxU, cdx, ndxtx)


    if int_select<9: 
        sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(int_select)    

        y = odeint(func=Xflow,y0=y0,t=time_range,args = (beta,theta,phim,sigma, rho, kappa, kappanr, SH, SM, HM, spec))    
        incnew2, incretx2, incinhnew2, incinhretx2, incmdrnew2, incmdrretx2, inctbhiv2, TBmort2, TBprev2, HIVprev2 = evalepi( y[-1,:], beta, theta, phim )

        print " "
        print "Incidence, new:", np.around( 1e5*incnew2, decimals=1 ), "per 100,000"
        print "Incidence, retx:", np.around( 1e5*incretx2, decimals=1), "per 100,000"
        print "Incidence, total:", np.around( 1e5*(incnew2 + incretx2), decimals=1 ), "per 100,000"
        print np.around((1-(incnew2 + incretx2)/(incnew + incretx))*100,decimals=1),"% reduction"
        print "Incidence, INH new:", np.around( 1e2*incinhnew2/incnew2, decimals=1 ), "%"
        print "Incidence, INH retx:", np.around( 1e2*incinhretx2/incretx2, decimals=1 ), "%"
        print "Incidence, MDR new:", np.around( 1e2*incmdrnew2/incnew2, decimals=2 ), "%"
        print "Incidence, MDR retx:", np.around( 1e2*incmdrretx2/incretx2, decimals=2 ), "%"
        print "Incidence, MDR total:", np.around( 1e5*(incmdrretx2+incmdrnew2) , decimals=2), "per 100,000"
        print np.around((1-(incmdrretx2+incmdrnew2)/(incmdrretx+incmdrnew))*100,decimals=1),"% reduction"
        print "Incidence, TB/HIV:", np.around( 1e2*inctbhiv2/(incnew2+incretx2), decimals=1 ), "%"
        print "TB mortality:", np.around( 1e5*TBmort2, decimals=1), "per 100,000"
        print np.around((1-(TBmort2)/(TBmort))*100,decimals=1),"% reduction"
        print "TB duration:", np.around( TBprev2/(incnew2+incretx2), decimals=2 ), "years"
        print "HIV prevalence:", np.around( 1e2*HIVprev2, decimals=1 ), "%"
        costvec = np.array( [ evalcost(y[i,:],sigma, kappa, costtxS, costtxU, cdx, ndxtx) for i in range(y.shape[0]) ] )
        c12 = sum(costvec[0:100])/100
        c52 = sum(costvec)/100
        print "Cost by end Year 1: $", np.around( c12 )
        print np.around( 1e2*(c12/c1-1),decimals=1),"% increase"
        print "Cost by end Year 5: $", np.around( c52 )
        if c52 > c5:
            print np.around( 1e2*(c52/c5-1),decimals=1),"% increase"
        if c52<=c5:
            print np.around( -1e2*(c52/c5-1), decimals=1),"% decrease"
        print " "

    impactarray = np.zeros(126)
    costvect3 = np.zeros(500)
    incvect3 = np.zeros(500)
    incredn5 = np.zeros(9)
    mdrredn5 = np.zeros(9)
    cmore5 = np.zeros(9)
    if int_select==9:
        for abc in range(9):
            intopt = abc
            sigma, rho, kappa, kappanr, SH, SM, HM, spec, costtxS, costtxU, cdx, ndxtx  = setparameters(intopt)    
            y = odeint(func=Xflow,y0=y0,t=time_range,args = (beta,theta,phim,sigma, rho, kappa, kappanr, SH, SM, HM, spec))    
            incnew2, incretx2, incinhnew2, incinhretx2, incmdrnew2, incmdrretx2, inctbhiv2, TBmort2, TBprev2, HIVprev2 = evalepi( y[-1,:], beta, theta, phim )

            print "INTERVENTION ",intopt
            print "Incidence, new:", np.around( 1e5*incnew2, decimals=1 ), "per 100,000"
            print "Incidence, retx:", np.around( 1e5*incretx2, decimals=1), "per 100,000"
            print "Incidence, total:", np.around( 1e5*(incnew2 + incretx2), decimals=1 ), "per 100,000"
            print np.around((1-(incnew2 + incretx2)/(incnew + incretx))*100,decimals=1),"% reduction"
            print "Incidence, INH new:", np.around( 1e2*incinhnew2/incnew2, decimals=1 ), "%"
            print "Incidence, INH retx:", np.around( 1e2*incinhretx2/incretx2, decimals=1 ), "%"
            print "Incidence, MDR new:", np.around( 1e2*incmdrnew2/incnew2, decimals=2 ), "%"
            print "Incidence, MDR retx:", np.around( 1e2*incmdrretx2/incretx2, decimals=2 ), "%"
            print "Incidence, MDR total:", np.around( 1e5*(incmdrretx2+incmdrnew2) , decimals=2), "per 100,000"
            print np.around((1-(incmdrretx2+incmdrnew2)/(incmdrretx+incmdrnew))*100,decimals=1),"% reduction"
            print "Incidence, TB/HIV:", np.around( 1e2*inctbhiv2/(incnew2+incretx2), decimals=1 ), "%"
            print "TB mortality:", np.around( 1e5*TBmort2, decimals=1), "per 100,000"
            print np.around((1-(TBmort2)/(TBmort))*100,decimals=1),"% reduction"
            print "TB duration:", np.around( TBprev2/(incnew2+incretx2), decimals=2 ), "years"
            print "HIV prevalence:", np.around( 1e2*HIVprev2, decimals=1 ), "%"
            costvec = np.array( [ evalcost(y[i,:],sigma, kappa, costtxS, costtxU, cdx, ndxtx) for i in range(y.shape[0]) ] )
            c12 = sum(costvec[0:100])/100
            c52 = sum(costvec)/100
            print "Cost by end Year 1: $", np.around( c12 )
            print np.around( 1e2*(c12/c1-1),decimals=1),"% increase"
            print "Cost by end Year 5: $", np.around( c52 )
            if c52 > c5:
                print np.around( 1e2*(c52/c5-1),decimals=1),"% increase"
            if c52<=c5:
                print np.around( -1e2*(c52/c5-1), decimals=1),"% decrease"
            print " "

            incredn5[abc] =  (1-(incnew2 + incretx2)/(incnew + incretx))*100
            mdrredn5[abc]= (1-(incmdrretx2+incmdrnew2)/(incmdrretx+incmdrnew))*100
            # ipm = np.array( [incprevmort(row) for row in OUT] ) # calculate costs etc over all 500 t
            # costvect3 = ipm[:,10]
            # incvect3 = ipm[:,0] + ipm[:,1]            
            # costvect2 = costvect3[1:100] # only timestep 0 to 99
            print " "
            cmore5[abc] = 1e2*(c52/c5-1)
            impactarray[abc*14:abc*14+11] = incnew2, incretx2, incinhnew2, incinhretx2, incmdrnew2, incmdrretx2, inctbhiv2, TBmort2, TBprev2, HIVprev2, c52
            impactarray[abc*14+11] = 0#sum(costvect2[:])/100
            impactarray[abc*14+12] = 0#sum(costvect3[:])/100
            impactarray[abc*14+13] = 0#sum(incvect3[:])/100
        # todo - reinstate these and file-saving
        

    if int_select == 9:
        print 'Graphing...'
        from matplotlib import pyplot as plt
        plt.close('all')
        plt.plot(incredn5,cmore5,marker='D',linestyle='None')
        plt.ylabel('% increase in cumulative costs by 5')
        plt.xlabel('% decrease in TB incidence at year 5')
        plt.axhline(color='r')
        plt.axvline(color='r')
        for i in range(9):
            dx,dy = 0.5*np.random.rand(),0.5*np.random.rand()
            plt.text(incredn5[i]+dy,cmore5[i]+dx,i)
        fn = os.path.join(os.path.dirname(__file__),"AllGraph1.pdf")
        plt.savefig(fn)
        print 'Graph 1 saved as AllGraph1.pdf'
        plt.close('all')
        plt.plot(mdrredn5,cmore5,marker='D',linestyle='None')
        plt.ylabel('% increase in cumulative costs by 5')
        plt.xlabel('% decrease in MDR-TB incidence at year 5')
        plt.axhline(color='r')
        plt.axvline(color='r')
        for i in range(9):
            dx,dy = 0.5*np.random.rand(),0.5*np.random.rand()
            plt.text(mdrredn5[i]+dy,cmore5[i]+dx,i)
        fn = os.path.join(os.path.dirname(__file__),"AllGraph2.pdf")
        plt.savefig(fn)
        print 'Graph 2 saved as AllGraph2.pdf'

        fn = os.path.join(os.path.dirname(__file__),"ImpactData.csv")        
        with open(fn,'wb') as f:
            writer=csv.writer(f)
            writer.writerow(impactarray[0:14]) 
            writer.writerow(impactarray[14:28])
            writer.writerow(impactarray[28:42])
            writer.writerow(impactarray[42:56])
            writer.writerow(impactarray[56:70])
            writer.writerow(impactarray[70:84])
            writer.writerow(impactarray[84:98])
            writer.writerow(impactarray[98:112])
            writer.writerow(impactarray[112:126])
        
    return y
# end of runmodel() function
        
# runmodel( 250, 0.83, 3.7 / 100, 500, 9 )         # run model for testing

############################################################
# GUI CLASSES
############################################################

class Checkbar(Frame):
    def __init__(self, parent=None,picks=[],side=TOP,anchor=W,label=''):
        Frame.__init__(self,parent)
        if label != '':
            Label(self, text=label).pack(side=TOP,anchor=N,expand=YES)
        self.vars = []
        for pick in picks:
            var = IntVar()
            chk = Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)
        self.config(relief=SUNKEN,bd=2) # borders around things
    def state(self):
        return [var.get() for var in self.vars]

class Radiobar(Frame):
    def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
        Frame.__init__(self,parent)
        self.var = StringVar()
        for pick in picks:
            rad = Radiobutton(self, text=pick, value=pick, variable=self.var)
            rad.pack(side=side, anchor=anchor, expand=YES)
        self.config(relief=SUNKEN,bd=2) # border
    def state(self):
        return self.var.get()

class Quitter(Frame):
    """ Generic class for quitting nicely, from PP"""
    def __init__(self,parent=None):
        Frame.__init__(self,parent)
        self.pack()
        widget = Button(self, text='Quit', command=self.quit)
        widget.pack(side=LEFT)
    def quit(self):
        ans = askokcancel('Verify exit',"Really quit?")
        if ans: Frame.quit(self)
            

class STDText(Frame):
    """A scollbar text class for redirection of output. Constructed with an initial message."""
    def __init__(self,parent=None,initial=''):
        Frame.__init__(self,parent)
        self.config(relief=SUNKEN,bd=2) # borders around things
        self.makewidgets(initial)
    def makewidgets(self,initial):
        sbar = Scrollbar(self)
        text = Text(self, relief=SUNKEN)
        sbar.config(command=text.yview)
        text.config(yscrollcommand=sbar.set)
        sbar.pack(side=RIGHT,fill=Y)
        text.pack(side=LEFT,expand=YES,fill=BOTH)
        self.text = text
        if initial!='':
            self.write(initial)
    def write(self, stuff):
        self.text.insert("end", stuff)
        self.text.yview_pickplace("end")

        
class Floatform(Frame):
    """A derived class for entering floats."""
    def __init__(self, parent=None, fields=[], defaults=[],label='',side=TOP,anchor=N):
        Frame.__init__(self,parent)    
        self.config(relief=SUNKEN,bd=2) # borders around things
        if label != '':
            Label(self, text=label ).pack(side=side,anchor=anchor,expand=YES)
        self.vars = []
        if len(defaults)!= len(fields): defaults = ['']*len(fields)
        for field,default in zip(fields,defaults):
            var = StringVar() 
            row = Frame(self)
            lab = Label(row,  text=field)
            ent = Entry(row)
            row.pack(side=TOP, fill=X)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.config(textvariable=var)
            ent.insert(0,default)
            self.vars.append(var)
    def state(self):        
        """Returns converted to floats"""
        return [float(var.get()) for var in self.vars]            


class Filechooser(Frame):
    """ Have one box toggling use, text and dialog.
    clabel - is the text for a checkbox toggling use, '' or default omits.
    blabel - is the text for the chooser button.
    save   - determines whether this is a save-as, or a open dialog.
    """
    def __init__(self, parent=None, side=TOP, anchor=W, clabel='', blabel='', save=True ):
        Frame.__init__(self,parent)
        self.config(relief=SUNKEN,bd=2) # borders around things
        self.vars = ['']*2
        self.var = IntVar()
        self.check = False      # checkbox?
        if clabel != '':        # checkbox construction
            chk = Checkbutton(self, text=clabel, variable=self.var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.activated = False # default not activated with checkbox present
            self.check = True
        else:
            self.activated = True # activated if not checkbox
            self.vars[0] = 1 # if no checkbox, set use to yes
        # chooser button
        widget = Button(self, text=blabel, command=(lambda: self.onPress(save)))
        widget.pack(side=LEFT)
    def refresh( self ):
        if self.check:      # must be var get, not sure if accessible
            self.vars[0] = self.var.get()
            self.activated = True if self.vars[0]!=0 else False    
    def onPress( self, saveFlag):
        self.refresh()
        if self.activated:
            if saveFlag:
                filename = asksaveasfilename()
            else:            
                filename = askopenfilename()
            self.vars[1] = filename
        else:
            print 'This action is not activated. Is there an unchecked checkbox?'
            self.vars[1] = ''
            print self.vars
        return self.vars
    def state(self):        
        """Returns converted to floats"""
        self.refresh()
        return self.vars

        
def makegridscale( frame ):
    """Make all the boxes in grid expand evenly."""
    ncr = frame.grid_size()     # cols,rows oddly
    for j in range(ncr[0]):
        frame.columnconfigure(j,weight=1)
    for i in range(ncr[1]):
        frame.rowconfigure(i,weight=1)        

#############################################################
# GUI CODE
#############################################################


if __name__ == '__main__':
    root = Tk()
    root.title('FlexDx-TB')

    # text entry - scenario
    fields = ['scenario']
    defaultfields=[0]
    sc = Floatform(root, fields=fields, defaults=defaultfields, label="Choose your diagnostic scenario:")
    sc.grid(column=0,row=0,sticky=NSEW)
    
    # text entry - targets
    vb = Floatform(root, fields=['TB Incidence, 100ky','HIV incidence, %/y','MDR, % in new TB','Cost drug 1, USD'], defaults=[250,0.83,3.7,500], label="Choose parameters:")
    vb.grid(column=0,row=1,sticky=NSEW)
    
    # execute
    def fitandrun( ):
        vbz = vb.state()
        target_inc = vbz[0]
        target_hiv = vbz[1]
        target_mdr = vbz[2]
        cost = vbz[3]
        print '\nUsing targets:\n\
         target_inc='+str(target_inc)+', target_hiv='+str(target_hiv)+', target_mdr='+str(target_mdr)+', cost='+str(cost)
        nsc = sc.state()
        int_select = nsc[0]
        if int_select >=0 and int_select<=9:
            print 'Running model for option', int_select ,'...'
            global OUT                    # want to persist for saving
            OUT = runmodel( target_inc, target_hiv, target_mdr / 100, cost, int_select )         # run
            print '...done.'
        else:
            print 'Currently only scenarios 0 to 9 are defined!'
            print 'You chose ', int(int_select), ' Ignoring...'
    rmbtn = Button(root,text='run model',command=(lambda: fitandrun()))
    rmbtn.grid(column=0,row=2,sticky=NSEW)       


    # text redirect box
    # todo: recheck these
    intxt = 'Welcome to FlexDx-TB v0.2!\n\
    Output will appear in this box.\n\
    Intervention options:\n\
    \t0 = baseline\n\
    \t1 = culture if previously treated\n\
    \t2 = Xpert for all\n\
    \t3 = MODS/TLA\n\
    \t4 = same-day smear\n\
    \t5 = same-day Xpert\n\
    \t6 = Xpert for smear-positive only\n\
    \t7 = Xpert for HIV-positive only\n\
    \t8 = Xpert with culture DST confirmation\n\
    \t9 = all (takes several times longer...saves graphs & csv file)\n'
    sTxt = STDText(root,initial=intxt)
    sTxt.grid(column=1,row=0,rowspan=5,columnspan=2,sticky=NSEW) # new col

    # return and quitter
    q = Quitter(root)
    q.grid(column=2)                    # 5

    # resizeability
    makegridscale( root )
    
    # redirect stdout
    sys.stdout = sTxt          # redirect
    sys.stderr = sTxt          # redirect

    # main
    root.mainloop()

