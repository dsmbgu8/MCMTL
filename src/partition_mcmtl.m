function [Xtr,ytr,Xte,yte] = partition_mcmtl(X,y,holdout,verbose)
% Input: multiclass per-domain data
% X = cell array of NT domains (e.g., source/target, etc.), dims NT x 1
% X{i} = matrix of samples from domain i, dims Ni x n (Ni=num samples, n=num features)
% y = cell array of multiclass labels for each task, dims NT x 1
% y{i} = array of multiclass labels for task i, dims Ni x 1
% holdout = percentage of samples to hold out for testing
% Output: training and test samples
% Xtr = cell array of training samples (of same form as X)
% ytr = cell array of training labels (of same form as y)
% Xte = cell array of test samples (of same form as X)
% yte = cell array of test labels (of same form as y)


if ~exist('verbose','var'), verbose=0; end;
if ~exist('holdout','var'), holdout=0.5; end;
randomize=1;

D = numel(X);
Xtr = {}; Xte = {};
ytr = {}; yte = {};
for Dj=1:D
  XDj = X{Dj};
  yDj = y{Dj};
  
  if randomize
    perm = randperm(numel(yDj));
    XDj = XDj(perm,:);
    yDj = yDj(perm);
  end
  
  cvp = cvpartition(yDj,'holdout',holdout);
  
  trIdx = cvp.training(1); 
  teIdx = cvp.test(1);
  
  
  Xtr = [Xtr XDj(trIdx,:)];
  ytr = [ytr yDj(trIdx)];
  Xte = [Xte XDj(teIdx,:)];
  yte = [yte yDj(teIdx)];   
  
  if verbose
    ntr = sum(trIdx); nte = sum(teIdx);    
    fprintf('Domain %d tr (%d): ',Dj,ntr); fprintf('%d ', unique(yDj(trIdx))); fprintf('\n') 
    fprintf('          te (%d): ',nte); fprintf('%d ', unique(yDj(teIdx))); fprintf('\n') 
  end
end