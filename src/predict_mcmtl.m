function [ypred,yscore] = predict_mcmtl(T,X,yuniq,scorefn,verbose)
% Input: task models, multiclass per-domain data
% Tasks = struct containing binary tasks from each domain, and their 
%     corresponding learning models and parameters (computed in learn_mcmtl.m)
% X = cell array of NT domains (e.g., source/target, etc.), dims NT x 1
% X{i} = matrix of samples from domain i, dims Ni x n (Ni=num samples, n=num features)
% yuniq = exhaustive set of K unique labels
% scorefn = binary scoring function of the form described in mcmtl.m
% Output: 

if ~exist('verbose','var'), verbose=0; end

D = numel(X);
K = numel(yuniq);
Ntask = numel(T);

ypred = {};
yscore = {};
for di=1:D
  Xdi = X{di};
  Ndi = size(Xdi,1);
  yscoredi = zeros([Ndi,K]);
  if Ntask == K % 1vsRest: pick most confident score of the predictors
    for yti=1:K
      % compute score for corresponding (positive) class
      yposti = T{yti}.ypos;
      yscoredi(:,yposti) = scorefn(T{yti}.model,Xdi,di);
      if verbose
        fprintf('Class %d min score: %0.3f, max score: %0.3f\n',...
                yposti,min(yscoredi(:,yposti)),max(yscoredi(:,yposti)));
      end
    end
  elseif Ntask == (K*(K-1))/2 % 1vs1: prediction by majority vote
    for yti = 1:Ntask      
      % increment pos/neg positions of corresponding classes
      yposti = T{yti}.ypos;
      ynegti = T{yti}.yneg;
      ysignti = sign(scorefn(T{yti}.model,Xdi,di));
      posidx = find(ysignti==1);
      negidx = find(ysignti==-1);
      yscoredi(posidx,yposti)=yscoredi(posidx,yposti)+1;
      yscoredi(negidx,ynegti)=yscoredi(negidx,ynegti)+1;
    end
  end
  [bestscore,bestidx] = max(yscoredi,[],2);
  ypred{di} = yuniq(bestidx); 
  yscore{di} = bestscore;
end