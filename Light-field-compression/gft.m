function [ sbar, V ] = gft( s, A )
%GFT Summary of this function goes here
%   Detailed explanation goes here

%     if cond(A)>1e3
%         [V,~] = eig(A);
%     else
%         [V,~] = jordan2(A);
%     end
    
%     [V,~] = jordan2(A);
    [V,~] = eig(A);
    sbar = inv(V)*s;

    


end

