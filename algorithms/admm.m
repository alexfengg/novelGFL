function [x,obj] = admm(y,lambda,rho,graph0,graph1,iter_num)
% graph0 is E0, graph1 is E1, where E0 has no two edges share one vertex
[n,p]=size(y);
l=size(graph0,1); % numebr of edges of E1
m=size(graph1,1); % numebr of edges of E1
x = cell(1,iter_num+1);
x(:) = {zeros(n,p)};
z = cell(1,iter_num+1);
z(:) = {zeros(m,p)};
u=z;
obj=zeros(1,iter_num);
vet=setdiff(union(graph0,graph1),graph0);

% index1 = cell(1,max(graph0(:)));
% index2 = cell(1,max(graph0(:)));

index1 = cell(1,max( max(graph0(:)),max(graph1(:))     ));
index2 = cell(1,max( max(graph0(:)),max(graph1(:))     ));

for i=1:l
    index1{graph0(i,1)} = find(graph1(:,1)==graph0(i,1));
    index1{graph0(i,2)} = find(graph1(:,1)==graph0(i,2));
    index2{graph0(i,1)} = find(graph1(:,2)==graph0(i,1));
    index2{graph0(i,2)} = find(graph1(:,2)==graph0(i,2));
end
if isempty(vet) == 0
    for i=1:length(vet)
        index1{vet(i)} = find(graph1(:,1)==vet(i));
        index2{vet(i)} = find(graph1(:,2)==vet(i));
    end
end


for j=1:iter_num
    
    % update for (x_s,x_t) for any edge (s,t) in E0
    
    ts=zeros(l,p);tt=zeros(l,p);ds=zeros(l,1);dt=zeros(l,1);
    for i=1:l
        [ts(i,:),ds(i,1)] = ft(graph0(i,1),j,index1,index2,graph1,x,u,z,rho);
        [tt(i,:),dt(i,1)] = ft(graph0(i,2),j,index1,index2,graph1,x,u,z,rho);
    end
    [A,B]=function2((2*y(graph0(:,1),:)-ts)./(2*(1+rho*ds)),(2*y(graph0(:,2),:)-tt)./(2*(1+rho*dt)),1+rho*ds,1+rho*dt,lambda);
    x{j+1}(graph0(:,1),:) = A;
    x{j+1}(graph0(:,2),:) = B;
    
    % update for E1
    % vectice that beloing in V but not in E0
    if isempty(vet) == 0
        t1 = zeros(length(vet),p); deg = zeros(length(vet),1);
        for i=1:length(vet)
            [t1(i,:),deg(i,1)] = ft(vet(i),j,index1,index2,graph1,x,u,z,rho);
        end
        x{j+1}(vet,:) = (2*y(vet,:)-t1)./(2*(1+rho*deg));
    end
    
    % update for z_st
    
    z{j+1}=threshold(x{j+1}(graph1(:,1),:)-x{j+1}(graph1(:,2),:)-u{j}/rho,lambda/rho);
    
    % update for u_st
    
    u{j+1}=u{j}+rho*(z{j+1}-x{j+1}(graph1(:,1),:)+x{j+1}(graph1(:,2),:));
    
    obj(j)=norm(x{j+1}-y,'fro')^2+...
        lambda*sum((sum((x{j+1}(graph0(:,1),:)-x{j+1}(graph0(:,2),:)).^2,2)).^0.5)+...
        lambda*sum((sum((x{j+1}(graph1(:,1),:)-x{j+1}(graph1(:,2),:)).^2,2)).^0.5);
end

end

function [x,y]=function2(a,b,c1,c2,w)
k = size(a,1);
for i=1:k
    if norm(a(i,:)-b(i,:),2) <= (c1(i,1)+c2(i,1))*w/(2*c1(i,1)*c2(i,1))
        x(i,:)=(c1(i,1)*a(i,:)+c2(i,1)*b(i,:))/(c1(i,1)+c2(i,1));
        y(i,:)=x(i,:);
    else
        x(i,:)=a(i,:)-w*(a(i,:)-b(i,:))/(2*c1(i,1)*norm(a(i,:)-b(i,:),2));
        y(i,:)=b(i,:)-w*(b(i,:)-a(i,:))/(2*c2(i,1)*norm(b(i,:)-a(i,:),2));
    end
end
end

function [x]=threshold(a,b)
[n,p]=size(a);
norm_a=sum(a.^2,2).^0.5;
weight=max(norm_a-b,0)./(norm_a+eps);
x=repmat(weight,1,p).*a;
end

function [t,d] = ft(i,k,index1,index2,graph1,x,u,z,rho)

index_1=index1{i};
index_2=index2{i};


t = -sum(u{k}(index_1,:)+rho*z{k}(index_1,:)+rho*x{k}(graph1(index_1,1),:)...
    +rho*x{k}(graph1(index_1,2),:),1)+sum(u{k}(index_2,:)+rho*z{k}(index_2,:)...
    -rho*x{k}(graph1(index_2,1),:)-rho*x{k}(graph1(index_2,2),:),1);
d = length(index_1)+length(index_2);


end

