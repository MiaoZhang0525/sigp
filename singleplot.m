xs = (-5:0.01:5)'; ns = size(xs,1); % complete data
xt = (-3:0.5:3)'; nt = size(xt,1); % a subset of data point for training\
fillcol = [0.530000 0.810000 0.980000];

keps = 1;

ne = length(keps);

% Mean function and covariance function
m = inline('0.25*x.^2 - 2*sin(x) + 1');
K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
KM = K(xs,xs)+keps*eye(ns);
KT = K(xt,xt)+keps*eye(nt);
ft = m(xt) + chol(KT)'*randn(nt,1); % sampled y

hyp = struct();
meanfunc = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = log(ones(2,1));
covfunc = @covSEiso; hyp.cov = log(ones(2,1));
likfunc = @likGauss; hyp.lik = 0;
hyp = minimize(hyp, @gp, -300, @infExact, meanfunc, covfunc, likfunc, xt, ft);
[yp,ys2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xt, ft, xs);
ciplot(yp-2*sqrt(ys2),yp+2*sqrt(ys2),xs,fillcol);
hold on;
plot(xs,yp,'k-');
plot(xt,ft,'ro','MarkerFaceColor','red');
xlabel('x^{(i)}');
ylabel('f(\cdot)');
set(gca,'fontsize',16)
