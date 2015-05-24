function outparm = cv_log_malsar(X,y,inparm)
%% Learns binary model and computes hyperparameters via cross validation 
%%  for several MALSAR multitask learning algorithms using the logistic loss 
%% Input: binary per-domain data
%% X = cell array of NT domains (e.g., source/target, etc.), dims NT x 1
%% X{i} = matrix of samples from domain i, dims Ni x n (Ni=num samples, n=num features)
%% y = cell array of binary labels for each task, dims NT x 1
%% y{i} = array of binary labels for task i, dims Ni x 1
%% inparm = struct of input parameters including
%%          method - MALSAR algorithm to use
%%          nsplit - number of cross validation splits
%%          valset - percentage of data to use for validation set in each split
%%          regidx - pick regularization parameters based on:
%%                   all tasks (1:NT), src only (1) or tgt only (2:NT) tasks
%%          verbose - enable verbose output

T = numel(X); % number of tasks


if ~exist('inparm','var') | isempty(inparm)
  inparm = struct();
end

method = 'Lasso'; 
if isfield(inparm,'method')
  method = inparm.method;
end


nsplit = 2; % folds for inner-cv loop
if isfield(inparm,'nsplit')
  nsplit=inparm.nsplit;
end

valset = 0.5; % percentage of data to use for validation
if isfield(inparm,'valset')
  valset=inparm.valset; 
end


regidx = 1:T;               
if isfield(inparm,'regidx')
  regidx=inparm.regidx; 
end

verbose=0;
if isfield(inparm,'verbose')
  verbose = inparm.verbose;
end


rho1v = [0,10e-5,0.001,0.1,0.3,0.5]; % sparsity constraint
rho1v = unique([rho1v, 1-rho1v]);
%rho1v = [rho1v,10,100,1000,10000];
%rho1v = rho1v(end:-1:1); % start with the silly big values first
rho2v = [0]; % L2 norm constraint, only Lasso, L21, CASO
kv = [1]; % shared feature space dim, only CASO

maxIter=2000; % number of optimization iterations (alg specific vals below)
tol = 10^-15; % pick a small tolerance since overtraining not an issue
initval=2; % init W, c from data
tFlag = 1; % terminate early if objective doesn't change

if strcmp(method,'Lasso')
  Wcfun = @(Xi,yi,r1,r2,k,opts)Logistic_Lasso(...
      Xi, yi, r1, merge_struct(opts,struct('rho_L2',r2)));  
  parmstr = @(r1,r2,k0)sprintf('rho1=%f, rho2=%f',r1,r2);
  rho1v = rho1v; 
  rho2v = rho1v; 
  % slooooooow!
  tol = 10^-10; 
elseif strcmp(method,'L21')
  Wcfun = @(Xi,yi,r1,r2,k,opts)Logistic_L21(...
      Xi, yi, r1, merge_struct(opts,struct('rho_L2',r2)));  
  parmstr = @(r1,r2,k0)sprintf('rho1=%f, rho2=%f',r1,r2);
  rho1v = rho1v;% (1:3:end-1);
  rho2v = rho1v; 
elseif strcmp(method,'Trace')
  Wcfun = @(Xi,yi,r1,r2,k,opts)Logistic_Trace(...
      Xi, yi, r1, opts);  
  parmstr = @(r1,r2,k0)sprintf('rho1=%f',r1);
  %rho1v = floor(size(X{1},2)*linspace(0,1,11)); % rank constraint
elseif strcmp(method,'CASO')
  Wcfun = @(Xi,yi,r1,r2,k0,opts)Logistic_CASO(...
      Xi, yi, r1, r2, k0, opts); 
  parmstr = @(r1,r2,k0)sprintf('rho1=%f, rho2=%f, k=%d',r1,r2,k0);
  rho2v = rho1v((rho1v>0)); % dont want 0 in rho1 / rho2 for cASO
  rho1v = [rho2v(end-1:-1:1)];  
  kv = ceil(size(X{1},2)*linspace(1/size(X{1},2),1-(1/size(X{1},2)),10));
  kv = kv(1:2:end);
  tol = 10^-10;
end

% default parameters to pass to Wcfun
Wcdefaults = struct('init',initval,'tFlag',tFlag,'tol',tol,'maxIter',maxIter); 


ncomplete = 0;
nparam = numel(rho1v)*numel(rho2v)*numel(kv);

bestRho1 = rho1v(1);
bestRho2 = rho2v(1);
bestAcc = 0;

if verbose, tic; end;
for rho1=rho1v
  if verbose, fprintf('computing param %d of %d\n',ncomplete,nparam); end
  for rho2=rho2v
    for k=kv
      yacc = zeros([T,1]);
      Wcdefaults_old = Wcdefaults;
      for spliti = 1:nsplit
        [Xtr,ytr,Xte,yte] = partition_mcmtl(X,y,valset);
        [W, c, funcVal] = Wcfun(Xtr,ytr,rho1,rho2,k,Wcdefaults);  
        % set the solution as the next initial point. 
        % this is more efficient according to MALSAR manual
        Wcdefaults.init = 1; 
        Wcdefaults.W0 = W;
        Wcdefaults.C0 = c;
        for di=1:T
          ypred = sign(predict_malsar(struct('W',W,'c',c),Xte{di},di));          
          yacc(di) = yacc(di)+(sum(ypred==yte{di})/numel(ypred));
          if verbose==3 % really verbose!
            fprintf('T%d: +=%d -=%d, pred+=%d, pred-=%d\n',...
                    di, sum(yte{di}==1), sum(yte{di}==-1),...
                    sum(ypred==1), sum(ypred==-1))
          end
        end
      end
      yacc = yacc / nsplit;
      Wcdefaults = Wcdefaults_old;
      
      if verbose==2 
        fprintf('acc(%s)=',parmstr(rho1,rho2,k)); 
        fprintf('%0.3f ',yacc); fprintf('\n')
      end
      % select rho1,rho2 with best average task accuracy on regidx values    
      yacc = mean(yacc(regidx)); 
      if yacc > bestAcc
        bestAcc = yacc;
        bestRho1 = rho1;
        bestRho2 = rho2;
        bestK = k;       
      end
    end
  end
  ncomplete = ncomplete + numel(rho2v)*numel(kv);
end


[bestW, bestc, bestFuncVal] = Wcfun(X,y,bestRho1,bestRho2,bestK,Wcdefaults);  

if verbose
  exectime=toc;
  fprintf('Logistic_%s acc (%s): %0.4f (elapsed time: %0.3fs)\n',method,...
          parmstr(bestRho1,bestRho2,bestK),bestAcc,exectime);
end


outparm         = struct();
outparm.W       = bestW;
outparm.c       = bestc;
outparm.rho1    = bestRho1;
outparm.rho2    = bestRho2;
outparm.funcVal = bestFuncVal;
outparm.accTr   = bestAcc;

