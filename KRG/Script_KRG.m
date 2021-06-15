clear;
rng(3,'v5uniform');
noise_var = 0.0;

experiment = 'klima_rec';  %klima_rec, klima_pred, brain (add _train for train)
init_get_data;

% Ntr_list = [50:50:N]; %for large datasets only

kernel_bw = 2*(25^2);
number_of_runs = 100;
alpha = 1e-4;
beta = 1e-1;

sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta);
% sim_params = struct('M',M,'Ntr_list',Ntr_list,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);

NMSE_test = zeros(number_of_runs,length(Ntr_list));
rng(123,'v5uniform');
tic
for run =1:number_of_runs
    fprintf('run %i - ', run);
        
    permutindex = randperm(Ndata);
    R = R0(:,permutindex);
    T = T0(:,permutindex);
    
    R_test = R(:,end-Nts+1:end);
    T_test = T(:,end-Nts+1:end);

    ntr_index=0;
    for Ntr=Ntr_list
        
        train_indices = 1:Ntr;
        R_train = R(:,train_indices);
        T_train = T(:,train_indices);
        
        ones_Ntr = ones(Ntr,1);
        K = exp(  -( reshape(vecnorm( kron(R_train',ones_Ntr)-kron(ones_Ntr,R_train')  ,2,2).^2, Ntr,Ntr)  )/kernel_bw  );
        
        T_train = T_train + sqrt(noise_var)*randn(M,Ntr);

        % Creating model with best parameters and noisy training data
        [U,Lambda] = eig(eye(M)+beta*L);
        [V,Sigma] = eig(K);
        lambda = diag(Lambda);
        sigma = diag(Sigma);
        P = V'*T_train'*U;
        q = ((alpha+kron(lambda,sigma)).^(-1)).*P(:);
        Q = reshape(q,Ntr,M);
        Psi = V*Q*U';       
        
        %%%%% TESTING %%%%%%%
        Ktest = exp(  -( reshape(vecnorm( kron(R_test',ones_Ntr)-kron(ones_Nts,R_train')  ,2,2).^2, Ntr,Nts)  )/kernel_bw  );
        Y_test = Psi'*Ktest;
        
        ntr_index=ntr_index+1;
        NMSE_test(run,ntr_index)  = (norm(Y_test-T_test,'fro')^2/norm(T_test,'fro')^2);
    end
    toc
end
plot([0 Ntr_list], [0 10*log10(mean(NMSE_test,1))])
hold on