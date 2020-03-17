function [x,obj] = admm_NFL(y,lambda,rho,graph,iter_num)

% input y=y(n,p), y(n,:) is the data point in R^p

[n,p]=size(y);
m=size(graph,1);

obj=zeros(1,iter_num);
x = cell(1,iter_num+1);
x(:) = {zeros(n,p)};
v1 = cell(1,iter_num+1);
v1(:) = {zeros(m,p)};
v2=v1;w1=v1;w2=v1;


index1 = cell(1,n);
index2 = cell(1,n);
deg = zeros(n,1);

for i=1:n
    index1{i} = find(graph(:,1)==i);
    index2{i} = find(graph(:,2)==i);
    deg(i,1) = length(index1{i})+length(index2{i});
end

for j=1:iter_num
    %     update for x
    temp = zeros(n,p);
    for k=1:n
        temp(k,:) =  sum(w1{j}(index1{k},:),1)+rho*sum(v1{j}(index1{k},:),1)+sum(w2{j}(index2{k},:),1)+rho*sum(v2{j}(index2{k},:),1);
    end
    x{j+1}=(2*y+temp)./(2+rho*deg);
    
    % update for v1(s,t),v2(s,t)
    [v1{j+1},v2{j+1}]=function1(x{j+1}(graph(:,1),:)-1/rho*w1{j},x{j+1}(graph(:,2),:)-1/rho*w2{j},2*lambda/rho);
    
    % update for w1(s,t),w2(s,t)
    w1{j+1}=w1{j}+rho*(v1{j+1}-x{j+1}(graph(:,1),:));
    w2{j+1}=w2{j}+rho*(v2{j+1}-x{j+1}(graph(:,2),:));
    
    
    obj(j)=norm(x{j+1}-y,'fro')^2+lambda*sum((sum((x{j+1}(graph(:,1),:)-x{j+1}(graph(:,2),:)).^2,2)).^0.5);
    
    
end

end



function [x,y]=function1(a,b,w)
% argmin (x-a)^2+(y-b)^2+w|x-y|
x = (a+b)/2+threshold((a-b)/2,w/2);
y = (a+b)/2-threshold((a-b)/2,w/2);
end

function [x]=threshold(a,b)
[n,p]=size(a);
norm_a=sum(a.^2,2).^0.5;
weight=max(norm_a-b,0)./(norm_a+eps);
x=repmat(weight,1,p).*a;
end


