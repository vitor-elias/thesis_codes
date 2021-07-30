function [ Xremoved ] = removeKlast( X, K )
%REMOVEKLAST Summary of this function goes here
%   Detailed explanation goes here

    if ~K
        Xremoved = X;
        return
    end

    Anor = X(:);
    Aabs = abs(X(:));
    
    [~,i] = sort(Aabs);
    
    s = Anor(i);
    
    s(1:K) = 0;
    
    Aquant(i) = s;
    
    Xremoved = reshape(Aquant,size(X));
    
    
    

end

