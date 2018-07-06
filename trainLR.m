function res = trainLR(X,y,m,k)

lambda = optimizableVariable('lambda', [1e-5,1e3], 'Transform', 'log');
cband  = optimizableVariable('cband',  [1e-1,1e3], 'Transform', 'log');

minfn = @(z) objfn(k,X,y,m,z.lambda,z.cband);

res = bayesopt(minfn,[lambda cband],'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'ExplorationRatio',0.5,'MaxObjectiveEvaluations',50);

end

function loss = objfn(k,X,y,m,lambda,cband)
c = cvpartition(length(y),'kFold',k);
fun = @(xT,yT,xt,yt) crossnlp(xT,yT,xt,yt,m,lambda,cband);
nlp = crossval(fun,X,y,'partition',c);
nlp = nlp(:);
loss = mean(nlp);
end

function nlp = crossnlp(xT,yT,xt,yt,m,lambda,cband)
hyp = sigp(xT,yT,m,'efn','lin',...
    'meankfn','sigp_lin','meankpar',[],...
    'covkfn', 'sigp_rbf','covkpar',cband,...
    'lambda',lambda,'normalize',false);
nlp = norm(yt-hyp.f(xt))/sqrt(size(yt,1));
%[yp,ys2] = hyp.f(xt);
%nlp = log(2*pi)/2 + sum(log(ys2)/2 + (yp-yt).^2./ys2/2)/size(yp,1);
end
