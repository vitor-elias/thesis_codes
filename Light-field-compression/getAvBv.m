function [ AvBv ] = getAvBv( bloco )
%GETAVBV Summary of this function goes here
%   Detailed explanation goes here

    X = bloco;
    
    A = zeros(length(X(:)));
    
    N1 = size(X,1);
    N2 = size(X,2);
    
    Ak = zeros(N1,N1-1,N2);
    Bk = zeros(N1,N2-1,N2);
    
    for k2 = 1:N2
        for i = 1:N1
            
            if i<N1
                Ak(i,i,k2) = X(i+1,k2);            
                Ak(i+1,i,k2) = X(i,k2);
            end
            
            if k2>1
                Bk(i,k2-1,k2) = X(i,k2-1);
            end
            
            if k2<N2
                Bk(i,k2,k2) = X(i,k2+1);
            end
        end
    end
    
    Av = [];
    Bv = [];
    
    for j = 1:N2
        Av = [Av;Ak(:,:,j)];
        Bv = [Bv;Bk(:,:,j)];
    end
    
    AvBv = [Av Bv];


end

