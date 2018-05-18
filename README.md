# sigp
Subspace-Induced Gaussian Processes - A dual Gaussian Process regression model over functions in the reproducing kernel Hilbert space. 

Data and code for our paper "Subspace-Induced Gaussian Processes": https://arxiv.org/pdf/1802.07528.pdf

![alt text](https://github.com/ZilongTan/sigp/blob/master/Example2.jpg "GP vs SIGP Comparison")

To generate the figure, run Example2.m in Matlab (you will need the [GPML Toolkit](http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html)).
This figure compares the predictive distribution given by the Gaussian Process (GP) and Subspace-Induced Gaussian Process (SIGP) using the training data points shown as red dots. The mean and covariance functions are respectively m(x)=x\*sin(x) and k(x,z) = exp(-5*(x-z)^2). The rows correspond respectively to noise variance 2, 1, and 0.5. The blue regions are the 2-sigma confidence intervals. As can be seen, SIGP better recovers the mean as well as uncertainty of the real data distribution.

If you like this project, consider citing the paper using the following BibTex entry:
```
@article{Tan18,
  author    = {Zilong Tan and Sayan Mukherjee},
  title     = {{Subspace-Induced Gaussian Processes}},
  journal   = {CoRR},
  volume    = {abs/1802.07528},
  year      = {2018}
  url = {https://arxiv.org/pdf/1802.07528.pdf}
}
```

## Example: Classification on ARCENE dataset

```matlab
disp("Loading the data ...");
feaTrain = load('data/arcene_train.data');
feaTest  = load('data/arcene_valid.data');
gndTrain = load('data/arcene_train.labels');
gndTest  = load('data/arcene_valid.labels');

disp("Standardizing the data ...");
fea = [feaTrain; feaTest];
fea = fea - mean(fea);
fea = fea ./ max(std(fea),1e-12);
feaTrain = fea(1:100,:);
feaTest  = fea(101:200,:);

disp("Classifying with SIGP ...");
hyp = sigp(feaTrain,gndTrain,1,'efn','ker','kparam',197,'lambda',1e-6);

disp("F1 score:" + num2str(F1score(sign(hyp.f(feaTest)),gndTest)));
```

In Matlab:
```
>> Example
Loading the data ...
Standardizing the data ...
Classifying with SIGP ...
F1 score:0.85417
```

### Fitting the Kernel Parameters using Cross-Validation
One way to select the kernel is to use the cross-validation. The example script trainKernel.m performs cross-validation and Baysian optimization for this task. First, edit the the range of the kernel parameters in the script, and then

In Matlab:
```matlab
res = trainKernel(X,y,1,5);
```
X,y are the regression feature matrix and response. The other parameters specify a rank-1 SIGP and 5 CV paritions to use. 
The kernel parameters can also be learned using the marginal likelihood.
