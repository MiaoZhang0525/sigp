xs = (-5:0.01:5)'; ns = size(xs,1); keps = [1e-1;1e-3;1e-6]; % complete data
xt = (-5:0.5:5)'; nt = size(xt,1); % a subset of data point for training\
fillcol = [0.530000 0.810000 0.980000];

ne = length(keps);

for i = 1:ne
    % Mean function and covariance function
    m = inline('0.25*x.^2');
    K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
    KM = K(xs,xs)+keps(i)*eye(ns);
    KT = K(xt,xt)+keps(i)*eye(nt);
    ft = m(xt) + chol(KT)'*randn(nt,1); % sampled y
    subplot(ne,3,1+3*(i-1));
    ciplot(m(xs)-2*sqrt(diag((KM))), m(xs)+2*sqrt(diag(KM)),xs,fillcol);
    ylim([-4,10]);
    hold on;
    plot(xs,m(xs),'k-','linewidth',2);
    plot(xt,ft,'ks');
    if i == 1, title('Real Data Dist'); end
    xlabel('x');
    ylabel('y');

    subplot(ne,3,2+3*(i-1));
    hyp = struct();
    meanfunc = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = log(ones(2,1));
    covfunc = @covSEiso; hyp.cov = log(ones(2,1));
    likfunc = @likGauss; hyp.lik = 0;
    hyp = minimize(hyp, @gp, -300, @infExact, meanfunc, covfunc, likfunc, xt, ft);
    [yp,ys2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xt, ft, xs);
    ciplot(yp-2*sqrt(ys2),yp+2*sqrt(ys2),xs,fillcol);
    ylim([-4,10]);
    hold on;
    plot(xs,yp,'k-');
    if i == 1, title('GP Predictive Dist'); end
    xlabel('x');
    ylabel('y');

    subplot(ne,3,3+3*(i-1));
    hyp = sigp(xt,ft,1,'kparam',21.885,'efn','ker','lambda',0.045767);
    [yp,ys2] = hyp.f(xs);
    ciplot(yp-2*sqrt(ys2),yp+2*sqrt(ys2),xs,fillcol);
    ylim([-4,10]);
    hold on;
    plot(xs,yp,'k-');
    if i == 1, title('SIGP Predictive Dist'); end
    xlabel('x');
    ylabel('y');
end
