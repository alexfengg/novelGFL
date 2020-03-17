clc
clear

% This is the test with Network Lasso algorithm and proposed alogrithm
% (Algorithm 1) in R^2, the graph is {(1,2),(1,3),(3,4)}. y\in R^{4*p}, p
% is the dimension of the data points. 

rng(1);
lambda=0.5;
rho=1;
iter_num=1000;


y=rand(4,2);
graph=[1 2;1 3; 3 4];
graph0=[1 2;3 4];
graph1=[1 3];

[x1 obj_alone] = admm(y,lambda,rho,graph0,graph1,10000);

allrho=2.^[0:1:6];
iter_num=1000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];

tic;
for i=1:length(allrho)
[x1 obj1{i}] = admm(y,lambda,allrho(i),graph0,graph1,iter_num);
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


%Network Lasso
[x2 obj_net] = admm_NFL(y,lambda,rho,graph,10000); 

tic;
for i=1:length(allrho)
[x2 obj2{i}] = admm_NFL(y,lambda,allrho(i),graph,iter_num);
end
toc

figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj2{i}-obj_net(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Network Lasso Algorithm','FontSize',16)


% figure;
% semilogy(1:iter_num,abs(obj1(end)*ones(1,iter_num)- obj1))
% hold on;
% semilogy(1:iter_num,abs(obj2(end)*ones(1,iter_num)- obj2))



