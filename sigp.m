function hyp = sigp(X,y,m,varargin)
% Subspace-Induced Gaussian Processes.
%
% Input:
%    X, y are the n-by-p feature matrix and n-by-1 label vector.
%    m specifies the rank of the desired RKHS
%
% Optional:
%    MaxIter is the max number of EM iterations
%    tol specifies the minimum change in the objective value between two
%        consecutive EM iterations.
%    ns  is the maximum number of slices, each slice corresponds to a range of y.
%        For classification, a slice contains one or more classes
%    eta is small postive number used to improve the condition of A
%    tau is used for linear mean function with p > n
%    lambda is the mean function regularization parameter
%    efn specifies the error function, 'lin' - linear, 'ker' - kernel Ridge
%    kfn gives the kernel function, default is RBF
%    kpar is the kfn parameter. For builtin RBF, kpar is the bandwidth
%    SDR - specifies whether to use SDR initialization, default is true
%
% Returns the model struct hyp. Some important members are
%    hyp.f  is the fitted target function f: X -> [Y,varY]
%    hyp.mf is the fitted mean function mf: X -> Y
%    hyp.nlp is a vector of negative log likelihood
%
% Copyright (c) 2018 Zilong Tan (ztan@cs.duke.edu)

hyp = struct();
n = size(X,1);
p = size(X,2);

opt = inputParser;
opt.addParameter( 'MaxIter', 50, @(x) floor(x) > 0 );
opt.addParameter( 'tol', 1e-4, @(x) floor(x) >= 0);
opt.addParameter( 'ns', max(m+1,floor(n/10)), @(x) floor(x) > 0 & floor(x) < n/2);
opt.addParameter( 'eta', 1e-8, @(x) floor(x) >= 0);
opt.addParameter( 'tau', 1e-8, @(x) floor(x) >= 0);
opt.addParameter( 'lambda', 1e-2, @(x) floor(x) >= 0);
opt.addParameter( 'efn', 'ker', @(x) strcmp(x,'lin')|strcmp(x,'ker'));
opt.addParameter( 'kfn', @(varargin) sigp_rbf(varargin{:}), @(x) x());
opt.addParameter( 'kparam', 1, @(x) true);
opt.addParameter( 'SDR', true, @(x) islogical(x));
opt.parse(varargin{:});
opt = opt.Results;

% Center the data (do not normalize the variance)
% Need adjustment for prediction
hyp.Xmu = mean(X);
X = X - hyp.Xmu;

if strcmp(opt.efn,'lin')
    use_ker_efn = false;
else
    use_ker_efn = true;
end

% Partition the data by response range
% This works for both regression and classification
[y,idx] = sort(y,'ascend');
X = X(idx,:);
[~,nun] = unique(y);
nun = [nun(2:end)-nun(1:end-1); n+1-nun(end)];
if length(nun) <= opt.ns
    csz = nun;
else
    csz(1,1) = 0;
    sz = n/opt.ns;
    i = 1;
    for j = 1:length(nun)
        if csz(i,1) >= sz
            i = i + 1;
            csz(i,1) = 0;
        end
        csz(i,1) = csz(i,1) + nun(j);
    end
end

K = opt.kfn(X,[],opt.kparam);
% Compute SDR matrices
A = zeros(n,n);
pos = cumsum([1;csz]);
for i = 1:length(csz)
    idx = pos(i):pos(i+1)-1;
    A(idx,:) = K(idx,:) - mean(K(idx,:));
end
C = K - mean(K);  C = C'*C;
A = A'*A;  A(1:n+1:end) = A(1:n+1:end) + opt.eta;

if opt.SDR
    % Initialize W with the SDR basis
    [W,D] = eig(C,A);
    [~,idx] = sort(diag(D),'descend');
    W = W(:,idx(1:m));
else
    % Initialize W randomly
    W = randn(n,m);
end

if use_ker_efn
    efn = @(varargin) sigp_efn_ker(y,K,varargin{:});
else
    efn = @(varargin) sigp_efn_lin(y,X,varargin{:});
end

% Initialize other parameters
s2 = 1;  Sb = eye(m); iSb = Sb;
beta = zeros(m,1);
err = y;  res = err;
P = K*W;  PTP = P'*P;
hyp.W = W./sqrt(sum(W.^2)); hyp.nlp = [];

for i = 1:opt.MaxIter
    % Use square root for better numerical conditions
    V = compV(P*sqrtm(Sb),s2);
    % Negative log-likelihood
    nlp = (log(2*pi)*(n+m) + pdlogdet(Sb) + n*log(s2) + ...
           beta'*iSb*beta + sum((res/sqrt(s2)).^2))/2/n;
    hyp.nlp = [hyp.nlp; nlp];
    % Fit mean function
    alp = efn(V,opt.lambda,opt.tau);
    err = efn(alp);
    if length(hyp.nlp) > 1 && hyp.nlp(end-1) - hyp.nlp(end) < opt.tol
        break;
    end
    Sv = inv(PTP/s2 + iSb);
    %beta = Sv*P'*err/s2;
    beta = Sb*P'*V*err;
    % Fit function variance
    Sb = beta*beta' + Sv;
    iSb = inv(Sb);
    res = err - P*beta;
    % Fit W and noise variance
    s2 = s2 + (sum(res.^2) - s2^2*trace(V))/n;
end

hyp.alpha = alp;
hyp.beta  = beta;
hyp.Q  = W*sqrtm(Sb);
hyp.SigmaNoise = s2;

MF = W*beta;
CF = W*sqrtm(Sv);

if strcmp(opt.efn,'ker')
    hyp.f = @(Z) sigp_pred_ker(opt.kfn(Z-hyp.Xmu,X,opt.kparam), ...
            @(KZ) -sigp_efn_ker(zeros(size(Z,1),1),KZ,alp), ...
            MF,CF,s2);
    hyp.mf = @(Z)-sigp_efn_ker(zeros(size(Z,1),1),...
             opt.kfn(Z-hyp.Xmu,X,opt.kparam),alp);
else
    hyp.f = @(Z) sigp_pred_lin(opt.kfn(Z-hyp.Xmu,X,opt.kparam), ...
            -sigp_efn_lin(zeros(size(Z,1),1),Z-hyp.Xmu,alp), ...
            MF,CF,s2);
    hyp.mf = @(Z)-sigp_efn_lin(zeros(size(Z,1),1),...
             Z-hyp.Xmu,alp);
end

hyp.kfn = @(Z) opt.kfn(Z-hyp.Xmu,X,opt.kparam);

end

function [pmu,pvar] = sigp_pred_ker(KZ,mufn,MF,CF,s2)
pmu = mufn(KZ) + KZ*MF; % use one reference to KZ to save computation
pvar = sum((KZ*CF).^2,2) + s2;
end

function [pmu,pvar] = sigp_pred_lin(KZ,muZ,MF,CF,s2)
pmu = muZ + KZ*MF;
pvar = sum((KZ*CF).^2,2) + s2;
end

% Linear mean function
function val = sigp_efn_lin(y,X,V,lambda,tau)
if nargin < 3, val = 'p+1'; return, end
if nargin == 3, val = y - X*V(2:end) - V(1); return, end
n = size(X,1);
p = size(X,2);
rs = sum(V);
ss = sum(sum(V));
VL = V - rs'/ss*rs;
if p > n
    % Dual estimator
    VL(1:n+1:end) = VL(1:n+1:end) + tau;
    val = X'*((X*X'+lambda*inv(VL))\y);
else
    CVL = X'*VL*X;
    CVL(1:p+1:end) = CVL(1:p+1:end) + lambda;
    val = CVL\(X'*VL*y);
end
val = [rs/ss*(y-X*val);val];
end

% Kernel Ridge mean function
function val = sigp_efn_ker(y,K,V,lambda,tau)
if nargin < 3, val = 'n+1'; return, end
if nargin == 3, val = y - K*V(2:end) - V(1); return, end
n = size(K,1);
rs = sum(V);
ss = sum(sum(V));
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

function K = sigp_rbf(X,Z,band)
if nargin < 1, K = 1; return, end
if nargin > 2
    X = X/band;
    if ~isempty(Z), Z = Z/band; end
end
if nargin < 2 || isempty(Z), Z = X; end

sqX = -sum(X.^2,2);
sqZ = -sum(Z.^2,2);
K = bsxfun(@plus, sqX, (2*X)*Z');
K = bsxfun(@plus, sqZ', K);
K = exp(K);

end
