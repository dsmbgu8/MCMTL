function Tasks = learn_mcmtl(X,y,yuniq,learnfn,mctype,balance,verbose)
% Input: multiclass per-domain data
% X = cell array of NT domains (e.g., source/target, etc.), dims NT x 1
% X{i} = matrix of samples from domain i, dims Ni x n (Ni=num samples, n=num features)
% y = cell array of multiclass labels \in [1,K] for each task, dims NT x 1
% y{i} = array of multiclass labels for task i, dims Ni x 1
% yuniq = exhaustive set of K unique labels
% learnfn = binary learning function of the form described in mcmtl.m
% mctype = {'1vs1', '1vsRest'}
% Output: 
% Tasks = struct containing binary tasks from each domain, and their 
%     corresponding learning models and parameters

if ~exist('mctype','var'), mctype='1vs1'; end
if ~exist('balance','var'), balance=0; end
if ~exist('verbose','var'), verbose=0; end

T = numel(X);
K = numel(yuniq);

Tasks = {}; 
Ti = 0;

for yi = 1:K
  if strcmp(mctype,'1vs1') % one vs one
    % create a task for each pair of classes in this domain
    for yj = yi+1:K
      Xp = {}; yp = {};
      for ti = 1:T
        Xti = X{ti};
        yti = y{ti};
        
        yidx = find(yti==yuniq(yi));  
        Xtii = Xti(yidx,:);
        yjdx = find(yti==yuniq(yj));  
        Xtij = Xti(yjdx,:);

        Xpn = [Xtii; Xtij];
        ypn = [ones([1,size(Xtii,1)]) -1*ones([1,size(Xtij,1)])]';
        if balance
          balance_idx = balance_classes(ypn);
          if numel(balance_idx) > 0
            Xpn = [Xpn; Xpn(balance_idx,:)];
            ypn = [ypn; ypn(balance_idx)];
          end
        end
        
        Xp = [Xp; Xpn];
        yp = [yp; ypn];
      end
      Ti = Ti+1;      
      Tasks{Ti} = struct('X',{Xp},'y',{yp},'ypos',yuniq(yi),'yneg',yuniq(yj));                  
      tic;
      Tasks{Ti}.model = learnfn(Xp,yp,[]);
      traintime=toc;
      if verbose
        fprintf('Training accuracy, class %d vs. %d: %0.4f (training time: %0.3fs)\n',...
                yuniq(yi),yuniq(yj),Tasks{Ti}.model.accTr,traintime);
      end
    end
  elseif strcmp(mctype,'1vsRest')
    % create a task for each class vs. the other classes in this domain
    Xp = {}; yp = {};
    for ti = 1:T
      Xti = X{ti};
      yti = y{ti};
      
      yidx = find(yti==yuniq(yi));
      Xtii = Xti(yidx,:);
      yjdx = find(yti~=yuniq(yi));
      Xtij = Xti(yjdx,:);
      
      Xpn = [Xtii; Xtij];
      ypn = [ones([1,size(Xtii,1)]) -1*ones([1,size(Xtij,1)])]';
      if balance
        balance_idx = balance_classes(ypn);
        Xpn = [Xpn; Xpn(balance_idx,:)];
        ypn = [ypn; ypn(balance_idx)];
      end
      
      Xp = [Xp; Xpn];
      yp = [yp; ypn];      
    end
    Ti = Ti+1;    
    Tasks{Ti} = struct('X',{Xp},'y',{yp},'ypos',yuniq(yi),'yneg',-yuniq(yi));                
    Tasks{Ti}.model = learnfn(Xp,yp,[]);
    if verbose
      fprintf('Training accuracy, class %d vs. rest: %0.4f\n',...
              yuniq(yi),Tasks{Ti}.model.accTr);
    end      
  end    
end

if verbose
  fprintf('Defined %d binary tasks (%s) for %d class classification problem\n',...
          Ti,mctype,K);
end
% Tasks{i}.X = cell array of NT domain samples, classes ypos vs yneg samples
% Tasks{i}.y = cell array of NT domain labels in [1,-1], classes ypos vs yneg samples
% Tasks{i}.ypos = label in [1,K] for Tasks{i}.y == 1
% Tasks{i}.yneg = label in [1,K] for Tasks{i}.y == -1, yneg=-ypos for 1vsRest 


