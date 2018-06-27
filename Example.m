% Example usage for the classification of Arcene data

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
%hyp = sigp(feaTrain,gndTrain,1,'efn','ker','kparam',321.25,'lambda',0.026811);
hyp = sigp(feaTrain,gndTrain,2,'efn','ker','kparam',331.73,'lambda',0.99);

disp("F1 score:" + num2str(F1score(sign(hyp.f(feaTest)),gndTest)));
