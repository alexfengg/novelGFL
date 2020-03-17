function [ graph ] = decomp_graph(m,n)
% This function is used to generate an m*n size 2D grid graph
%   [graph] = decomp_graph(m,n) returns 4 cells of edges, in which 
%   there are no edges sharing one vertex

if ~exist('n','var')
    n=m;
end

if m<2|n<2
    disp('Please enter some numbers > 2!')
    return;
end

graph = cell(1,4);

temp1 = [(1:2:n-1)',(1:2:n-1)'+1];
temp2 = [(2:2:n-1)',(2:2:n-1)'+1];
temp3 = [(1:2*n:(m-1)*n)',(1:2*n:(m-1)*n)'+n];
temp4 = [(1+n:2*n:(m-1)*n)',(1+n:2*n:(m-1)*n)'+n];

for i = 1:m
    graph{1} = [graph{1};temp1+(i-1)*n];
    graph{2} = [graph{2};temp2+(i-1)*n];
end

for j=1:n
    graph{3} = [graph{3};temp3+j-1];
    graph{4} = [graph{4};temp4+j-1];
end


end

