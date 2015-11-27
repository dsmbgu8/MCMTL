# Matlab MultiClass MultiTask Learning (MCMTL) toolbox. 

## Summary 

This toolbox provides MATLAB code to decompose a multi-class, multi-task learning problem into a set of 1vs1 or 1vsRest binary, multi-task subproblems, which are combined to generate multi-class, multi-task predictions via majority vote (1vs1) or maximum likelihood (1vsRest). 

Example learning and prediction functions are provided for the MALSAR (Multi-tAsk Learning via StructurAl Regularization) toolbox in the src/MALSAR directory. The MALSAR toolbox is available at: http://www.public.asu.edu/~jye02/Software/MALSAR/


## System Requirements 

Tested using Matlab 7.14.0.739 (R2012a) on a Macbook Pro running OSX 10.6.

## Installation 

Run the following code to add the MCMTL functions to your path:
```matlab
MCMTL_ROOT='/path/to/MCMTL/';
addpath(genpath(MARTIAL_ROOT));
savepath; % optional 
  ```
  
## Example Usage 

Data: given tasks Ti, each an [nTi x n] matrix, with [nTi x 1] vector labels TLi, i \in [0,2], generate predictions for task T0 and T3 using training data from tasks T0 and T1.  
```matlab
mtlXtr = {T0,T1};  % training tasks
mtlytr = {TL0,TL1}; % training task labels
mtlXte = {T0,T1,T2};  % test tasks
mtlyte = {TL0,TL1,TL2}; % test task labels
  ```
  
Define mcmtl parameters
```matlab
mcmtl_nfold=2;
mcmtl_balance=false; 
mcmtl_mctype='1vs1';
mcmtl_parms = struct('mctype',mcmtl_mctype,'balance',mcmtl_balance,'verbose',1,'nfold',mcmtl_nfold);
  ```
  
Set up MALSAR learning and prediction functions
```matlab
malsar_method = 'Trace'; % 'L21';  % use trace norm regularization
malsar_parm = struct('method',malsar_method,'verbose',0,'nsplit',1);
mcmtl_learnfn = @(Xp,yp,inparm)cv_log_malsar(Xp,yp,malsar_parm);
mcmtl_scorefn = @(Tmodel,Xdi,di)predict_malsar(Tmodel,Xdi,di);  
```
Generate predictions and print mean/stddev of cross-validated prediction accuracies for each task
```matlab
mcmtl_outparm=mcmtl(mtlXtr,mtlytr,mtlXte,mtlyte,mcmtl_learnfn,mcmtl_scorefn,mcmtl_parms);
acc_mcmtl = mcmtl_outparm.accT; % per-task accuracies
cvacc_mcmtl = mean(acc_mcmtl,1); % mean of accuracy across folds
cvstd_mcmtl = std(acc_mcmtl,0,1); % stddev of accuracy across folds
fprintf('T0: %0.4f (%0.4f) T1: %0.4f (%0.4f) T2: %0.4f (%0.4f)\n',...
             cvacc_mcmtl(1),cvstd_mcmtl(1),cvacc_mcmtl(2),cvstd_mcmtl(2),...
             cvacc_mcmtl(3),cvstd_mcmtl(3));
```

## Citation 

When using this code, please use the following citations below.

```bibtex
@phdthesis{bue2013,
  title        = {Adaptive Similarity Measures for Material Identification in Hyperspectral Imagery},
  author       = {Brian D. Bue},
  organization = {Rice University, Houston TX},
  address      = {\url{http://www.ece.rice.edu/~bdb1/#mcmtl}}
}

@manual{mcmtl2013,
  title        = {MultiClass MultiTask Learning (MCMTL) toolbox},
  author       = {Brian D. Bue},
  organization = {Rice University, Houston TX},
  address      = {\url{http://www.ece.rice.edu/~bdb1/#mcmtl}},
}
```


## Disclaimer 

Although this code has been reasonably well-tested, it is research code, and may contain bugs. Please feel free to contact the author (bbue@alumni.rice.edu) if you have any difficulties. 

## Changelog 

06/17/13 - initial release.

## Contact 

Please feel free to contact the author (bbue@alumni.rice.edu) if you have any questions or issues with the code described in this readme.

