clc
clear

% We run a Chicago crime data by using novelGFL and Network Fused Lasso
% alogrithm. More details about this example see, Efficient Implementations
% of the Generalized Lasso Dual Path Algorithm, TB Arnold, RJ Tibshirani, 2016


load edges
load input

y1 = edge_coloring(y); % color the edges
graph = edge_graph(y1); % forming the graphs, here total is 20
K = length(graph);

n = max(max(y)); % number of vertices
% a = rand(n,1); % given data

rho = 100;


graph0=graph{1};
graph1=[graph{2}];

for i=3:K
    graph1=[graph1;graph{i}];
end
graph_NFL = cell2mat(graph');

a = initialinput;
%% lambda=0.05

lambda = 0.05;

%[x1,obj1] = admm(a,lambda,rho,graph0,graph1,10000);


[x2,obj1] = admm_NFL(a,lambda,rho,graph_NFL,10000);


allrho=2.^[0:1:6];
iter_num=2000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x1 obj{i}] = admm(a,lambda,allrho(i),graph0,graph1,iter_num);
end

% Plot the graph
h1=figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj{i}-obj1(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Algorithm 1','FontSize',16)
axis([0,2000,10^-13,10^4])


for i=1:length(allrho)
    [x1 obj_NFL{i}] = admm_NFL(a,lambda,allrho(i),graph_NFL,iter_num);
end
% Plot the graph
h2=figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj_NFL{i}-obj1(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Network Lasso Algorithm','FontSize',16)
axis([0,2000,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
saveas(h1,'ChiCrime005_algorithm1.png')
saveas(h1,'ChiCrime005_algorithm1.eps')
saveas(h2,'ChiCrime005_NFL.png')
saveas(h2,'ChiCrime005_NFL.eps')


%% lambda=0.25

lambda = 0.25;

%[x1,obj1] = admm(a,lambda,rho,graph0,graph1,10000);

[x2,obj1] = admm_NFL(a,lambda,rho,graph_NFL,10000);

allrho=2.^[0:1:6];
iter_num=2000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x1 obj{i}] = admm(a,lambda,allrho(i),graph0,graph1,iter_num);
end

% Plot the graph
h3=figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj{i}-obj1(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Algorithm 1','FontSize',16)
axis([0,2000,10^-13,10^4])


for i=1:length(allrho)
    [x1 obj_NFL{i}] = admm_NFL(a,lambda,allrho(i),graph_NFL,iter_num);
end
% Plot the graph
h4=figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj_NFL{i}-obj1(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2^{0}','2^{1}','2^{2}','2^{3}','2^{4}','2^{5}','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Network Lasso Algorithm','FontSize',16)
axis([0,2000,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
saveas(h3,'ChiCrime025_algorithm1.png')
saveas(h3,'ChiCrime025_algorithm1.eps')
saveas(h4,'ChiCrime025_NFL.png')
saveas(h4,'ChiCrime025_NFL.eps')








