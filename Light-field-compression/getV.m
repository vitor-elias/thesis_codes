function [ V ] = getV( A )
%GETV Summary of this function goes here
%   Detailed explanation goes here


    
    
% % M�todo do Laplaciano
%     D = diag(sum(A));
%     L = D-A;
%     [V,~] = eig(L);
%     
    
    [V,~] = eig(A);


end

