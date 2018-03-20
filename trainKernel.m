function res = trainKernel(X,y,m,k)
% Simple code for choosing the kernel for SIGP using cross-validation.
% The fitting is based on the Bayesian Optimization toolbox of Matlab.
% Gradient-based methods are generally better.
%
% X, y are the data
% m is the rank of SIGP
% k specifies the number of CV partitions
%
% Copyright (c) 2018 Zilong Tan (ztan@cs.duke.edu)

band   = optimizableVariable('band', [1,1e3],'Transform','log');

minfn = @(z) objfn(k,X,y,m,z.band);

res = bayesopt(minfn,band,'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'ExplorationRatio',0.5,'MaxObjectiveEvaluations',30);

end

function loss = objfn(k,X,y,m,band)
c = cvpartition(y,'kFold',k);
fun = @(xT,yT,xt,yt) crossnlp(xT,yT,xt,yt,m,band);
nlp = crossval(fun,X,y,'partition',c);
nlp = nlp(:);
loss = mean(nlp);
end

function nlp = crossnlp(xT,yT,xt,yt,m,band)
hyp = sigp(xT,yT,m,'kparam',band,'efn','ker','lambda',1e-2);

% Use the L2 loss function (useful for classification)
nlp = norm(yt-hyp.f(xt))/sqrt(size(yt,1));

%Use the NLPD loss function (useful for regression)
%[yp,ys2] = hyp.f(xt);
%nlp = log(2*pi)/2 + sum(log(ys2)/2 + (yp-yt).^2./ys2/2)/size(yp,1);

end
