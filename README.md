# sigp
Subspace-Induced Gaussian Processes

Data and code for our paper "Subspace-Induced Gaussian Processes": https://arxiv.org/pdf/1802.07528.pdf

If you like this project, consider citing the paper using the following BibTex entry:
```
@article{Tan18,
  author    = {Zilong Tan and
               Sayan Mukherjee},
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
F1 score:0.86667
```
### Fitting the Kernel Parameters using Cross-Validation

In Matlab:
```matlab
res = trainKernel(X,y,1,5);
```
X,y are the regression feature matrix and response. The other parameters specify a rank-1 SIGP and 5 CV paritions to use.
