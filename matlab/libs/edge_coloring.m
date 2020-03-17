function [y] = edge_coloring(y)
% This is a realization of Greedy algorithm for coloring the edges
%   y is the edges, each row stands for an edge
%   [y] = edge_coloring(y) returns the edges with the corresponding colors


n=size(y,1);
y(:,3)=zeros(n,1);
for i=1:n
    index1=find(y(1:i-1,1)==y(i,1));%find the indeices of the edges whose left node is y(i,1)
    index2=find(y(1:i-1,1)==y(i,2));%..                                   right node is y(i,2)
    index3=find(y(1:i-1,2)==y(i,1));
    index4=find(y(1:i-1,2)==y(i,2));
    index=[index1;index2;index3;index4];% all the neighboring edges
    colors_to_avoid=y(index,3);% the colors of the neighboring edges
    % find the first available color
    j=1;
    while length(find(colors_to_avoid==j))>0
        j=j+1;
    end
    y(i,3)=j;
end

end