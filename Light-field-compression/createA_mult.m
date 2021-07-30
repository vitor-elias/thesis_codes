function [ A ] = createA_mult( K_blocos)
%CREATEA_MULT Summary of this function goes here
%   Detailed explanation goes here


    X = K_blocos;

    
    s = X(:,:,1);
    A = zeros(length(s(:)));
    
    N1 = size(s,1);
    N2 = size(s,2);
    
    %%%%%% SEM CVX %%%%%%%
    C = [];
    b = [];
    for i = 1:size(X,3)
        s = X(:,:,i);
        AvBv(:,:,i) = getAvBv(s);
        C = [C;AvBv(:,:,i)];
        b = [b;s(:)];
    end
    
    rho = 0;
    if rank(C)<min(size(C))
        rho = 1e-5;
    end


    ab = inv(C'*C + rho*eye(size(C'*C)))*C'*b;
    %%%%%%%%%%%%%%%%%%%%%%%     

    
%     %%%%%%%%%%%%%% CVX %%%%%%%%%%%%%%%%%
%     for i = 1:size(X,3)
%         AvBv(:,:,i) = getAvBv(X(:,:,i));
%     end
%     
%     cvx_begin quiet
%     
%         obj = 0;
%         variable ab(N1+N2-2)
%         
%         for i= 1:size(X,3)
%           s = X(:,:,i);
%           obj = obj + norm(AvBv(:,:,i)*ab - s(:));
%         end
%         
%         minimize(obj)     
%         
%     cvx_end
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    a = ab(1:N1-1);
    b = ab(N1:end);

    M = N1;
    N = N2;
    for i = 1:M
        for j = 1:N

            %Pixel i-1
            i2 = i-1;
            if i2 >= 1                
                A(M*(j-1)+i,M*(j-1)+i2) = a(i-1);
            end


            %Pixel i+1
            i2 = i+1;            
            if i2 <=M
                A(M*(j-1)+i,M*(j-1)+i2) = a(i);
            end

            %Pixel j-1
            j2 = j-1;            
            if j2 >= 1
                A(M*(j-1)+i,M*(j2-1)+i) = b(j-1);
            end

            %Pixel j+1
            j2 = j+1;

            if j2 <= N
                A(M*(j-1)+i,M*(j2-1)+i) = b(j);
            end

        end
    end

%     A = quant(A,0.05);
end

