function mcmtlparm = mcmtl(Xtr,ytr,Xte,yte,learnfn,scorefn,inparms)
% Decomposes a multi-class, multi-task problem into a set of binary multi-task subproblems that can be learned with a binary multi-task/domain adaptation algorithm. 
% Input:
%   Xtr - Cell array of dimension T containing training samples from each task
%         Each entry Xtr{i} is a Ntri x ni array, where Ntri is the number of 
%         training samples, each of dimensionality ni, for task i.
%   ytr - Cell array of dimension T containing training labels from each task
%         Each entry ytr{i} is a Ni x 1 array with values in the range {1,...,K}
%   Xte - Cell array of dimension T containing testing samples from each task                            
%         Each entry Xte{i} is a Ntei x ni array, where Ntei is the number of 
%         testing samples, each of dimensionality ni, for task i.        
%   yte - Cell array of dimension T containing testing labels from each task                          
%         Each entry Xte{i} is a Ntei x 1 array with values in same range as ytr
%   learnfn - binary learning function of the form Task.model = learn(Xtr,ytr,param), 
%             where
%             Xtr,ytr - T dimensional cell arrays as above, ytr in [-1,1]
%             param - input parameters to learning function
%             Tmodel - struct containing learned model params for binary task
%   scorefn - scoring function of the form 
%             yscoredi = score(Ti,Xti,ti), where
%             Task - Task parameters, including Task.model output params from learnfn
%             Xti - Nti x ni array of samples for task ti
%             ti - scalar task index for Xti in {1,...,D}
%             yscoreti - scores for each Xti in [-C,C], C \in reals 
% Output:
%   mcmtlparm - struct of output parameters including:
%               accT - T x 1 array with test accuracy for each task
%
% Assumptions:
% - Ntri, Ntei > 1 for all i in {1,...,D}

T = numel(Xtr); %number of tasks

if ~exist('inparms','var')
  inparms = struct();
end

nfold=2;
if isfield(inparms,'nfold')
  nfold = inparms.nfold;
end

mctype='1vs1';
if isfield(inparms,'mctype')
  mctype=inparms.mctype;
end

verbose=0;
if isfield(inparms,'verbose')
  verbose=inparms.verbose;
end

balance=0;
if isfield(inparms,'balance')
  balance=inparms.balance;
end

yuniq = [];
for Tj=1:T, yuniq = unique([yuniq; ytr{Tj}; yte{Tj}]); end

accT = zeros(nfold,T);
fprT = zeros(nfold,T);
fnrT = zeros(nfold,T);

if nfold == 1
  Ttr = learn_mcmtl(Xtr,ytr,yuniq,learnfn,mctype,balance);
  [ytepred,ytescore] = predict_mcmtl(Ttr,Xte,yuniq,scorefn);
  % compute per-task accuracy for each domain for this fold
  for Tj=1:T
    yteTj = yte{Tj};
    ytepredTj = ytepred{Tj};
    accT(1,Tj) = sum(yteTj==ytepredTj)/numel(yteTj);    
    cmatT = confusionmat(ytepredTj,yteTj);
    [fprj fnrj] = fprfnr(cmatT);
    fprT(1,Tj) = fprj;
    fnrT(1,Tj) = fnrj;
  end  
  if verbose
    fprintf('Per-task accuracy: '); 
    fprintf('%0.4f ',accT(1,:)); 
    fprintf('\n');
  end  
  mcmtlparm.ytepred = ytepred;
else
  % these give the percent of TEST points to hold out in training/test sampling
  train_holdout = 1-((nfold-1)/nfold); % e.g., nfold=3, 2/3 training, 1/3 test
  test_holdout = 1-train_holdout;

  for foldi=1:nfold
    if verbose, fprintf('Fold %d of %d...\n', foldi, nfold); end

    % split each task into train/test sets    
    [Xtri,ytri,ig,ig] = partition_mcmtl(Xtr,ytr,train_holdout);
    [Xtei,ytei,ig,ig] = partition_mcmtl(Xte,yte,test_holdout); 

    Ttr = learn_mcmtl(Xtri,ytri,yuniq,learnfn,mctype,balance);
    [yteipred,yteiscore] = predict_mcmtl(Ttr,Xtei,yuniq,scorefn);
    
    % compute per-task accuracy for each domain for this fold
    for Tj=1:T
      yteiTj = ytei{Tj};
      yteipredTj = yteipred{Tj};
      accT(foldi,Tj) = sum(yteiTj==yteipredTj)/numel(yteiTj);    
      cmatT = confusionmat(yteipredTj,yteiTj);
      [fprj fnrj] = fprfnr(cmatT);
      fprT(foldi,Tj) = fprj;
      fnrT(foldi,Tj) = fnrj;
    end  
    if verbose
      fprintf('Per-task accuracy: '); 
      fprintf('%0.4f ',accT(foldi,:)); 
      fprintf('\n');
    end
  end
end
mcmtlparm.inparms = inparms;
mcmtlparm.mctype = mctype;
mcmtlparm.balanced = balance;
mcmtlparm.nfold = nfold;
mcmtlparm.accT = accT; 
mcmtlparm.fprT = fprT; 
mcmtlparm.fnrT = fnrT; 
