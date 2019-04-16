function [ W ] = visibility_graph( T,directed )
%VISIBILITY_GRAPH Summary of this function goes here
%   T: nx2 times series. first column times, second intensities

n = size(T,1);
y = T(:,2);
t = T(:,1);
W = sparse(n,n);
%strcmp(parallel,'parallel')==1
%sort for divide and conquer algo
[~,id_sort] = sort(y,'descend');

for ii=1:n
    node1 = id_sort(ii);
    limit_right = node1-1+find(W(node1,node1:end)>0,1,'first');
    if isempty(limit_right)==1
        limit_right = n;
    end
    limit_left = find(W(node1,1:node1)>0,1,'last');
    if isempty(limit_left)==1
        limit_left = 1;
    end
    search_range_limit = (limit_left:1:limit_right)';
    for jj=1:length(search_range_limit)
        node2 = search_range_limit(jj);
        if node2 == node1+1
            w = compute_w(node1,node2,t,y);
            W(node1,node2) = w;
            W(node2,node1) = w;
        elseif node1 == node2+1
            w = compute_w(node1,node2,t,y);
            W(node1,node2) = w;
            W(node2,node1) = w;
        elseif node2~=node1
            a = find(search_range_limit==min(node1,node2));
            b = find(search_range_limit==max(node1,node2));
            idx = search_range_limit(a:b);
%             node1 = min(node1,node2);
%             node2 = max(node1,node2);
            idx(1) = []; idx(end) = [];
            d = y(node2)+(y(node1)-y(node2)).*(t(node2)-t(idx))./(t(node2)-t(node1));
            if sum(d>y(idx))==length(d)
                w = compute_w(node1,node2,t,y);
                W(node1,node2) = w;
                W(node2,node1) = w;
            end
        end
    end
end

if directed == 1
    for ii=1:n
        W(ii,1:ii) = zeros(1,ii);
    end
end
end

function out = compute_w(node1,node2,t,y)
%out = sqrt((t(node2)-t(node1))^2+(y(node2)-y(node1))^2);
%out = (t(node2)-t(node1))*(y(node2)-y(node1));
%out = abs((y(node2)-y(node1)))./(t(node2)-t(node1));
out = atan((y(node2)-y(node1))/(t(node2)-t(node1)))+pi/2;
%out = abs((t(node2)-t(node1)))*y(node2);
end