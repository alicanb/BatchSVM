function [labels,scores]=lssvmclassify(features,R,model)
% Classify samples using LSSVM
% Written by Kivanc Kose and Alican Bozkurt

%% INPUTS
%features = Feature Vector
% R = affinity matrix
%model = trained LSSVM model

m=size(features,1);
if model.theta~=0
    R=model.theta*sparse(R) + speye(m);
else
    R=speye(m);
end
scores=R*(features*model.w-ones(m,1)*model.gamma);
labels=sign(scores);