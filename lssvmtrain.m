function [model]=lssvmtrain(features,labels,R,nu,theta)
% Train linear locally smooth SVM classifier
% Written by Volkan Vural
% Updated by Kivanc Kose and Alican Bozkurt

%solves following problem:
%min(w;g;x;v) nu*e'*xi+e'*v
%s.t. D_j*[(theta*R_j+I)(B_j*w-gamma*e)]+xi_j>=e for j=1,..,k
%v >= w >= v
%xi >= 0:

%% INPUT
%features = Feature Vector
%labels = class labels
% R = affinity matrix
% nu = SVM parameter nu
% teta = SVM parameter theta

[m,n]=size(features);
D=spdiags(labels,0,m,m);
Rbar=theta*sparse(R) + speye(m);
e=ones(m,1);
%Rbar=Rbar/(4*teta+1);

%x=[w; gamma; v; xi]
A=[D*Rbar*features  -D*Rbar*e  sparse(m,n)   speye(m);... % D*Sbar*(Aw-gamma) + xi >= e
    speye(n)  sparse(n,1)  speye(n)   sparse(n,m);...             % w+s >= 0
    -speye(n) sparse(n,1)  speye(n)   sparse(n,m)];              % -w+s >= 0
b=[ones(m,1); zeros(2*n,1)];

%optimization parameters
f = [ zeros(1,n+1), ones(1,n), nu*ones(1,m)]; %nu*e'*xi+e'*v
lb= [-Inf*ones(1,n+1),  zeros(1,n+m)  ]; %xi>0
model.theta=theta;
model.w=0;
model.gamma=0;
option=optimset('linprog');option=optimset(option,'Display','off');
[x,~,model.converged] = linprog(f,-A,-b,[],[],lb,[],[],option);
model.converged=model.converged==1;
if model.converged==1
    model.w=x(1:n);     model.gamma=x(n+1);    %s=x(n+2:2*n+1);   %y=x(2*n+2:2*n+1+m);
end