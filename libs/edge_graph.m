function [graph] = edge_graph(y)
% This is a simple function that generate graphs for 
%   the Chicago Crime Data Experiment
%   [graph] = edge_graph(y) returns a groups of graphs
%   y(:,3) is the number of groups and the first two columns
%   are the edges

K = max(y(:,3)); % # of colors
graph = cell(1,K); 

for i = 1:K
    index = find(y(:,3)==i);
    graph{i}(:,1)=y(index,1);
    graph{i}(:,2)=y(index,2);
end

end