x = linspace(-10,10,300)';
z = log(abs(x)) + exp(sin(x));   % true response
y = z + z.*randn(length(x),1)/4; % noisy response

% Train Rank-3 SIGP using 25% of the data
% Use trainCR(x(1:5:end),y(1:5:end),3,2) to get the kernel parameters
hyp = sigp(x(1:4:end),y(1:4:end),3,'efn','cov',...
      'covkfn', 'sigp_rbf','covkpar',1.69396,...
      'normalize',false);
scatter(x,y,'bo');
hold on;
[ymu,ys2] = hyp.f(x);
plotshaded(x',[ymu'+2*sqrt(ys2');ymu';ymu'-2*sqrt(ys2')],'m','r-');
xlabel('x');
ylabel('y');
legend('data','prediction','95% Credible Interval');
set(gca,'fontsize',16);