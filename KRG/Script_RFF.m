clear;
rng(3,'v5uniform');
noise_var = 0;

experiment = 'brain';  %klima_rec, klima_pred, brain (add _train for train)
init_get_data;


% Ntr_list = [500:500:N]; %for large datasets only

kernel_bw = 2*(3^2);
% kernel_bw = 200;
number_of_runs = 10;

D = 32;
alpha = 1e-2;
beta = 1e-1;

% RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
Vrf = sqrt(kernel_bw^-1)*randn(input_dimension,D); % sampled vectors for RF
brf = 2*pi*rand(1,D); % random phase
Z0 = (D/2)^(-1/2)*cos(R0'*Vrf + brf);

sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta);
% sim_params = struct('M',M,'Ntr_list',Ntr_list,'Nts',Nts,'N',N,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);

NMSE_test = zeros(number_of_runs,length(Ntr_list));
rng(123,'v5uniform');
tic
for run = 1:number_of_runs
    fprintf('MC run %i of %i - ', run,number_of_runs);
    
    permutindex = randperm(Ndata);
    Z = Z0(permutindex,:);
    T = T0(:,permutindex);
    
    Z_test = Z(end-Nts+1:end,:);
    T_test = T(:,end-Nts+1:end);
   
    ntr_index=0;
    for Ntr=Ntr_list

        train_indices = 1:Ntr;        
        Z_train = Z(train_indices,:);
        T_train = T(:,train_indices);

        % Adding AWGN to the data
        T_train = T_train + sqrt(noise_var)*randn(size(T_train));

        % Creating model with best parameters and noisy training data
        ZT = Z_train'*T_train';

        [U,Lambda] = eig(eye(M)+beta*L);
        [V,Sigma] = eig(Z_train'*Z_train);
        lambda = diag(Lambda);
        sigma = diag(Sigma);
        P = V'*ZT*U;
        q = ((alpha+kron(lambda,sigma)).^(-1)).*P(:);
        Q = reshape(q,D,M);
        Psi = V*Q*U';    

        %%%%% TESTING %%%%%%%
        Y_test = Psi'*Z_test';
        
        ntr_index=ntr_index+1;
        NMSE_test(run,ntr_index) = (norm(Y_test-T_test,'fro')^2/norm(T_test,'fro')^2);
        
    end
    toc
end

% 10*log10(mean(NMSE_train))
plot([0 Ntr_list], [0 10*log10(mean(NMSE_test,1))])
hold on