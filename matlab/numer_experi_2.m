clear
rng(1);
n=16;
m=16;

% RGB fi
for k=1:3
    for i=1:n
        for j=1:n
            if (i-n/2)^2+(j-n/2)^2<(n/6)^2
                y(i,j,k)=0;
            else
                y(i,j,k)=0.3*k+0.1;
            end
        end
    end
end
y = imresize(y,[m m]); y = y+0.1*randn(m,m,3);
figure;
imshow(y)
for i=1:3
    x{i}=y(:,:,i);
    x{i}=x{i}(:);
end
x0=cell2mat(x);

% for i=1:n
%     for j=1:n
%         if (i-n/2)^2+(j-n/2)^2<(n/6)^2
%             y(i,j)=0;
%         else
%             y(i,j)=1;
%         end
%     end
% end
% 
% y = y+0.1*randn(m,m);
% x0 = y(:);

graph = decomp_graph(m);
graph_GFL = cell2mat(graph');
graph0 = graph{1};
graph1 = [graph{2};graph{3};graph{4}];
%% 1D experiment with lambda = 1
lambda=1;
tic;
[x1 obj_alone]=admm(x0,lambda,2,graph0,graph1,10000);
toc

allrho=2.^[0:1:6];
iter_num=2000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x1 obj1{i}] = admm(x0,lambda,allrho(i),graph0,graph1,iter_num);
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
axis([0,2000,10^-13,10^4])

%~~~~~~~~~~~~~Network Lasso~~~~~~~~~~~~~~~~~~~~~~~~~~
tic;
for i=1:length(allrho)
    [x1 obj2{i}] = admm_GFL(x0,lambda,allrho(i),graph_GFL,iter_num);
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
axis([0,2000,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Alogirhtm 1 takes %s second, Network Fused Lasso Algorithm takes %d second. Lambda=1.\n',t1,t2)
saveas(h1,[pwd '\figures\2Dlambda1_alogirhtm1.png'])
saveas(h2,[pwd '\figures\2Dlambda1_NFL.png'])


%% lambda=10;

lambda=10;


%[x2 obj_alone]=admm(x0,lambda,2,graph0,graph1,10000);
[x2 obj_alone]=admm_NFL(x0,lambda,2,graph_GFL,10000);



allrho=2.^[0:1:6];
iter_num=2000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];
tic;
for i=1:length(allrho)
    [x2 obj1{i}] = admm(x0,lambda,allrho(i),graph0,graph1,iter_num);
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
axis([0,2000,10^-13,10^4])

%~~~~~~~~~~~~~Network Lasso~~~~~~~~~~~~~~~~~~~~~~~~~~
tic;
for i=1:length(allrho)
    [x2 obj2{i}] = admm_NFL(x0,lambda,allrho(i),graph_GFL,iter_num);
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
axis([0,2000,10^-13,10^4])
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Alogirhtm 1 takes %s second, Network Fused Lasso Algorithm takes %d second. Lambda=10.\n',t1,t2)
saveas(h3,[pwd '\figures\2Dlambda10_alogirhtm1.png'])
saveas(h4,[pwd '\figures\2Dlambda10_NFL.png'])


%beta1=reshape(x2{end},[m m 3]);
%figure;
%imshow(beta2)

%% 
lambda=10;


%[x2 obj_alone]=admm(x0,lambda,2,graph0,graph1,10000);
[x2 obj_alone]=admm_NFL(x0,lambda,2,graph_GFL,10000);



allrho=[2:0.4:4];
iter_num=2000;
colors=[1,0,0;0,1,0;0,0,1;0.5,0,0;0,0.5,0;0,0,0.5;0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;1,1,0;0,1,1;1,0,1;0,0,0];


for i=1:length(allrho)
    [x2 obj1{i}] = admm(x0,lambda,allrho(i),graph0,graph1,iter_num);
end

% Plot the graph
h4=figure;
for i=1:length(allrho)
    semilogy(1:iter_num,obj1{i}-obj_alone(end),'Color',colors(i,:),'LineWidth',2);
    hold on;
end
h_legend=legend('\rho=2','2.4','2.8','3.2','3.6','4.0','2^{6}','Location','NorthEast');
set(h_legend,'FontSize',16);
ylabel('error','FontSize',16)
xlabel('number of iterations','FontSize',16)
set(gcf, 'OuterPosition', [100 100 500 450]);
title('Algorithm 1','FontSize',16)
axis([0,2000,10^-13,10^4])



