clc
clear

% This is the test with Network Lasso algorithm and proposed alogrithm
% (Algorithm 1) in R^2, the graph is 1D chain with n=100. y\in R^{4*p}, p
% is the dimension of the data points. 

rng(1);
lambda=2;
rho=1;
iter_num=1000;

n=16; % n is even
%graph{1}=[(1:2:n)' (2:2:n)'];
%graph{2}=[(2:2:n-1)' (3:2:n)'];

%graph1=[(1:n-1)' (2:n)'];

graph = decomp_graph(n);
graph1 = graph{1};
graph2 = [graph{2}; graph{3}; graph{4}];
graph_GFL = [graph1;graph2];

y=randn(n^2,2);


%[x1 obj_alone]=admm(y,lambda,rho,graph{1},graph{2},10000);
%[x2 obj_net]=admm_GFL(y,lambda,rho,graph1,10000);

tic;
[x1 obj_alone]=admm(y,lambda,rho,graph1,graph2,100);
toc

allrho=2.^[0:1:6];
iter_num=1000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];

tic;
for i=1:length(allrho)
%[x1 obj1{i}] = admm(y,lambda,allrho(i),graph{1},graph{2},iter_num);

[x1 obj1{i}] = admm(y,lambda,allrho(i),graph1,graph2,iter_num);
end
toc

figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj1{i}-obj_alone(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Algorithm 1','FontSize',16)
axis([0,1000,10^-12,10^4])

%Network Lasso

tic;
for i=1:length(allrho)
[x2 obj2{i}] = admm_NFL(y,lambda,allrho(i),graph_GFL,iter_num);
end
toc

figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj2{i}-obj_alone(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Network Lasso Algorithm','FontSize',16)
axis([0,1000,10^-12,10^4])
