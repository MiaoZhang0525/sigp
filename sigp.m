function hyp = sigp(X,y,m,varargin)
% Subspace-Induced Gaussian Processes.
%
% Input:
%    X, y are the n-by-p feature matrix and n-by-1 label vector.
%    m specifies the rank of the desired RKHS
%
% Optional:
%    maxiter is the max number of EM iterations
%    tol specifies the minimum change in the objective value between two
%        consecutive EM iterations.
%    ns  is the maximum number of slices, each slice corresponds to a range of y.
%        For classification, a slice contains one or more classes
%    eta is small postive number used to improve the condition of A
%    lambda is the mean function regularization parameter
%    efn specifies the error function, 
%        'cov' - use covariance kernel, 'lin' - linear, 'ker' - kernel Ridge
%    meankfn/covkfn give the mean and covariance kernel functions,
%       default are linear and rational quadratic
%
% Returns the model struct hyp. Some important members are
%    hyp.f  is the fitted target function f: X -> [Y,varY]
%    hyp.mf is the fitted mean function mf: X -> Y
%    hyp.nlp is a vector of negative log likelihood
%
% Copyright (c) 2018 Zilong Tan (ztan@cs.duke.edu)

hyp = struct();
[n,p] = size(X);

opt = inputParser;
opt.addParameter( 'maxiter',  50,        @(x) floor(x) > 0 );
opt.addParameter( 'tol',      1e-5,      @(x) floor(x) >= 0);
opt.addParameter( 'ns',       0,         @(x) floor(x) > 1 & floor(x) <= n/2); % 1 for auto select
opt.addParameter( 'eta',      1e-11,     @(x) floor(x) >= 0);
opt.addParameter( 'lambda',   1e-5,      @(x) floor(x) >= 0);
opt.addParameter( 'efn',      'cov',     @(x) strcmp(x,'lin')|strcmp(x,'ker')|strcmp(x,'cov'));
opt.addParameter( 'meankfn',  @sigp_lin, @(x) feval(x) >= 0); % used only if opt.efn == 'ker'
opt.addParameter( 'covkfn',   @sigp_rq,  @(x) feval(x) >= 0);
opt.addParameter( 'meankpar', [],        @(x) true);
opt.addParameter( 'covkpar',  [1 1],     @(x) true);
opt.addParameter( 'showlik',  false,     @(x) islogical(x));
opt.addParameter( 'normalize',true,      @(x) islogical(x));
opt.parse(varargin{:});
opt = opt.Results;    

hyp.opt = opt;

if strcmp(opt.efn,'lin')
    use_ker_efn = 0;
elseif strcmp(opt.efn,'cov')
    use_ker_efn = 1; % mean function uses the kernel as covariance
else
    use_ker_efn = 2; % for full kernel mean
end

% Center the data (do not normalize the variance)
% Need adjustment for prediction
if opt.normalize
    hyp.Xmu  = mean(X);
    hyp.Xstd = max(std(X)*sqrt(p),1e-12);
else
    hyp.Xmu  = zeros(1,size(X,2));
    hyp.Xstd = ones(1,size(X,2));
end
X = X - hyp.Xmu;
X = X ./ hyp.Xstd;

% Partition the data by response range
% This works for both regression and classification
[y,idx] = sort(y,'ascend');
X = X(idx,:);
[~,nun] = unique(y);
ns = opt.ns;
if ns == 0 % auto select
    ns = min(length(nun),floor(n/2));
end
nun = [nun(2:end)-nun(1:end-1); n+1-nun(end)];
if length(nun) <= ns
    csz = nun;
else
    csz(1,1) = 0;
    sz = n/ns;
    i = 1;
    for j = 1:length(nun)
        if csz(i,1) >= sz
            i = i + 1;
            csz(i,1) = 0;
        end
        csz(i,1) = csz(i,1) + nun(j);
    end
end

covkfn  = @(X,Z,param) feval(opt.covkfn,X,Z,param);
CK = covkfn(X,[],opt.covkpar);

if m > 0
    % Compute RKHS dimension reduction matrices
    A = zeros(n,n);
    pos = cumsum([1;csz]);
    for i = 1:length(csz)
        idx = pos(i):pos(i+1)-1;
        A(:,idx) = CK(:,idx) - mean(CK(:,idx),2);
    end
    C = CK; C = C - mean(C,2); C = C*C'/n;
    A = A*A'/n; A(1:n+1:end) = A(1:n+1:end) + opt.eta;
    hyp.A = A; hyp.C = C;
    [W,E] = eigs(C,A,m);
    hyp.eigs = diag(E);
else
    % use the full covariance kernel
    m = n;
    W = eye(n);
end

% Initialize other parameters
s2 = 1; Sb = eye(m); iSb = Sb;
beta = zeros(m,1);
err = y;  res = err;
P = CK*W;  PTP = P'*P;
hyp.nlp = [];

if use_ker_efn == 1
    efn = @(varargin) sigp_efn_cov(y,P,varargin{:});
elseif use_ker_efn == 2
    MK  = feval(opt.meankfn,X,[],opt.meankpar);
    efn = @(varargin) sigp_efn_ker(y,MK,varargin{:});
else
    efn = @(varargin) sigp_efn_lin(y,X,varargin{:});
end

if opt.showlik
    figure;
    hold on;
    title('Negative Log-Likelihood');
    xlabel('Iteration');
    ylabel('NLL');
end

for i = 1:opt.maxiter
    % Use square root for better numerical conditions
    V = compV(P*sqrtm(Sb),s2);
    % Negative log-likelihood
    nlp = (log(2*pi)*(n+m) + pdlogdet(Sb) + n*log(s2) + ...
           beta'*iSb*beta + sum((res/sqrt(s2)).^2))/2/n;
    hyp.nlp = [hyp.nlp; nlp];
    % Fit mean function
    alp = efn(V,opt.lambda);
    err = efn(alp);
    if opt.showlik, plot(hyp.nlp,'b-','linewidth',2); end
    if length(hyp.nlp) > 1 && hyp.nlp(end-1) - hyp.nlp(end) < opt.tol
        break;
    end
    Sv = inv(PTP/s2 + iSb);
    beta = Sv/s2*P'*err;
    Sb  = beta*beta' + Sv;
    iSb = inv(Sb);
    res = err - P*beta;
    % Fit the noise variance
    s2 = s2 + (sum(res.^2) - s2^2*trace(V))/n;
end

hyp.W = W;
hyp.alpha = alp;
hyp.beta  = beta;
hyp.Sb = Sb;
hyp.Proj = P;
hyp.NoiseVar = s2;

MF = W*beta;
CF = W*sqrtm(Sv);

hyp.covkfn = @(Z) covkfn((Z-hyp.Xmu)./hyp.Xstd,X,opt.covkpar);
if use_ker_efn == 1
    hyp.mf = @(Z)-sigp_efn_cov(zeros(size(Z,1),1),...
                covkfn((Z-hyp.Xmu)./hyp.Xstd,X,opt.covkpar)*W,alp);    
elseif use_ker_efn == 2
    hyp.mf = @(Z)-sigp_efn_ker(zeros(size(Z,1),1),...
                feval(opt.meankfn,(Z-hyp.Xmu)./hyp.Xstd,X,opt.meankpar),alp);    
else
    hyp.mf = @(Z)-sigp_efn_lin(zeros(size(Z,1),1),(Z-hyp.Xmu)./hyp.Xstd,alp);    
end

hyp.f = @(Z) sigp_pred(hyp.covkfn(Z),hyp.mf(Z),MF,CF,s2);

end

function [pmu,pvar] = sigp_pred(KZ,muZ,MF,CF,s2)
pmu  = muZ + KZ*MF;
pvar = sum((KZ*CF).^2,2) + s2;
end

% Linear mean function
function val = sigp_efn_lin(y,X,V,lambda)
if nargin < 3, val = 'p+1'; return, end
if nargin == 3, val = y - X*V(2:end) - V(1); return, end
n = size(X,1);
p = size(X,2);
rs = sum(V);
ss = sum(sum(V));
VL = V - rs'/ss*rs;
if p > n
    % Dual estimator
    XTVL = X'*VL;
    KVL = X*XTVL;
    KVL(1:n+1:end) = KVL(1:n+1:end) + lambda;
    val = XTVL/KVL*y;
else
    CVL = X'*VL*X;
    CVL(1:p+1:end) = CVL(1:p+1:end) + lambda;
    val = CVL\(X'*VL*y);
end
val = [rs/ss*(y-X*val);val];
end

% Mean function based on covariance kernel
function val = sigp_efn_cov(y,P,V,lambda)
if nargin < 3, val = 'm+1'; return, end
if nargin == 3, val = y - P*V(2:end) - V(1); return, end
rs = sum(V);
ss = sum(rs);
PTVL= P'*V - P'*rs'/ss*rs;
val = (PTVL*P)\(PTVL*y);
val = [rs/ss*(y-P*val); val];
end

% Kernel Ridge mean function
function val = sigp_efn_ker(y,K,V,lambda)
if nargin < 3, val = 'n+1'; return, end
if nargin == 3, val = y - K*V(2:end) - V(1); return, end
n = size(K,1);
rs = sum(V);
ss = sum(rs);
VL = V - rs'/ss*rs;
VLK = VL*K;
VLK(1:n+1:end) = VLK(1:n+1:end) + lambda;
val = VLK\(VL*y);
val = [rs/ss*(y-K*val); val];
end

function V = compV(P,s2)
n = size(P,1);
m = size(P,2);
V = P'*P;
V(1:m+1:end) = V(1:m+1:end) + s2;
V = -P/V*P';
V(1:n+1:end) = V(1:n+1:end) + 1;
V = V/s2;
end

function val = pdlogdet(X)
S = svd(X);
val = sum(log(S));
end

function K = sigp_lin(X,Z,param)
if nargin == 0 || isempty(X), K = 0; return, end
if nargin == 3 && isempty(Z), Z = X; end
K = X*Z';
end

function K = sigp_rbf(X,Z,band)
if nargin == 0 || isempty(X), K = 1; return, end
if nargin == 3
    X = X/band;
    if ~isempty(Z)
        Z = Z/band;
    else
        Z = X;
    end
end
sqX = -sum(X.^2,2);
sqZ = -sum(Z.^2,2);
K = bsxfun(@plus, sqX, (2*X)*Z');
K = bsxfun(@plus, sqZ', K);
K = exp(K);
end
