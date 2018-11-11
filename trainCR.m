function res = trainCR(X,y,m,k)

band  = optimizableVariable('band', [1e-2,1e2],'Transform', 'log');

minfn = @(z) objfn(k,X,y,m,z.band);

res = bayesopt(minfn,band,'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'ExplorationRatio',0.5,'MaxObjectiveEvaluations',30);

end

function loss = objfn(k,X,y,m,band)
c = cvpartition(length(y),'kFold',k);
fun = @(xT,yT,xt,yt) crossnlp(xT,yT,xt,yt,m,band);
nlp = crossval(fun,X,y,'partition',c);
nlp = nlp(:);
loss = mean(nlp);
end

function nlp = crossnlp(xT,yT,xt,yt,m,band)
hyp = sigp(xT,yT,m,'efn','cov',...
    'covkfn', 'sigp_rbf','covkpar',band,...
    'normalize',false);
nlp = norm(yt-hyp.f(xt))/sqrt(size(yt,1));
end
