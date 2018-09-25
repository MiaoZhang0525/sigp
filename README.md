# SIGP
Subspace-Induced Gaussian Processes - Gaussian processes induced by sufficient dimension reduction subspaces of the reproducing kernel Hilbert space.

## Comparison of SIGP and the standard Gaussian process
1. SIGP can express strictly a superset of functions realizable by GP for a fixed kernel
2. SIGP is computationally faster due to the low-rank of the covariance
3. SIGP is robust against overfitting (illustrated by the following example)

## Example: Classification of the ARCENE data

```matlab
disp("Loading the data ...");
feaTrain = load('data/arcene_train.data');
feaTest  = load('data/arcene_valid.data');
gndTrain = load('data/arcene_train.labels');
gndTest  = load('data/arcene_valid.labels');
% Standardizing the data
fea = [feaTrain;feaTest];
fea = fea - mean(fea);
fea = fea./max(std(fea),1e-12);
feaTrain = fea(1:100,:);
feaTest = fea(101:end,:);

disp("Classifying with SIGP ...");

hyp = sigp(feaTrain,gndTrain,1,'efn','lin',...
    'meankfn','sigp_lin','meankpar',[],...
    'covkfn', 'sigp_rbf','covkpar',0.017152,...
    'lambda',0.25265,'normalize',false);

disp("F1 score:" + num2str(F1score(sign(hyp.f(feaTest)),gndTest)));
```

In Matlab:
```
>> Example
Loading the data ...
Classifying with SIGP ...
F1 score:0.84783
```

### Fitting the Kernel Parameters using Cross-Validation
One way to select the kernel is to use the cross-validation. The example script trainLR.m combines cross-validation and Baysian optimization for this task:

In Matlab:
```matlab
res = trainLR(X,y,1,3);
```
X,y are the regression feature matrix and response. The other parameters specify a rank-1 SIGP and 3 CV paritions to use. 
The kernel parameters can also be learned using the marginal likelihood.

### Details
If you use SIGP in your applications, kindly consider citing the paper "Subspace-Induced Gaussian Processes": https://arxiv.org/pdf/1802.07528.pdf

