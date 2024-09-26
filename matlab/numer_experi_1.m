% admm.m is the Algorithm 1 of "A graph decomposition-based approach for the graph-fused lasso"

clear
rng(1);
y = zeros(100,2);
y(1:11,:)=1; 
y(12:22,1)=-1;
y(12:22,2)=1;
y(23:33,:)=2; 
y(34:44,:)=-1; 
y(45:100,:)=0; 



y=y+randn(100,2);
graph{1}=[(1:2:100)' (2:2:100)'];
graph{2}=[(2:2:99)' (3:2:100)'];



iter_num=1000;
%% 1D experiment with lambda = 1
lambda1=1;

[x1 obj_alone]=admm(y,lambda1,2,graph{1},graph{2},10000);

allrho=2.^[0:1:6];
iter_num=1000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x1 obj1{i}] = admm(y,lambda1,allrho(i),graph{1},graph{2},iter_num);
end
t1=toc;
% Plot the graph
h1=figure;
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
axis([0,400,10^-13,10^4])

%~~~~~~~~~~~~~Network Lasso~~~~~~~~~~~~~~~~~~~~~~~~~~
tic;
for i=1:length(allrho)
    [x2 obj2{i}] = admm_NFL(y,lambda1,allrho(i),[graph{1};graph{2}],iter_num);
end
t2=toc;
% Plot the graph
h2=figure;
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
axis([0,400,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Alogirhtm 1 takes %s second, Network Fused Lasso Algorithm takes %d second. Lambda=1.\n',t1,t2)
saveas(h1,[pwd '\figures\1Dlambda1_alogirhtm1.png'])
saveas(h2,[pwd '\figures\1Dlambda1_NFL.png'])

figure;
plot(1:100,y')
hold on
plot(1:100,x1{end}')
title('lambda=1')

%% lambda=10;

lambda1=10;
[x1 obj_alone]=admm1(y,lambda1,2,graph{1},graph{2},10000);

allrho=2.^[0:1:6];
iter_num=1000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x1 obj1{i}] = admm1(y,lambda1,allrho(i),graph{1},graph{2},iter_num);
end
t1=toc;
% Plot the graph
h3=figure;
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
axis([0,1000,10^-13,10^4])

%~~~~~~~~~~~~~Network Lasso~~~~~~~~~~~~~~~~~~~~~~~~~~
tic;
for i=1:length(allrho)
    [x2 obj2{i}] = admm_GFL(y,lambda1,allrho(i),[graph{1};graph{2}],iter_num);
end
t2=toc;
% Plot the graph
h4=figure;
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
axis([0,1000,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Alogirhtm 1 takes %s second, Network Fused Lasso Algorithm takes %d second. Lambda=10.\n',t1,t2)
saveas(h3,[pwd '\figures\1Dlambda10_alogirhtm1.png'])
saveas(h4,[pwd '\figures\1Dlambda10_NFL.png'])
%%lambda=10;

figure;
plot(1:100,y')
hold on
plot(1:100,x1{end}')
title('lambda=10')
