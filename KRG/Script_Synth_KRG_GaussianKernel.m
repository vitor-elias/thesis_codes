clearvars -except R500
rng(1,'v5uniform');

M = 50;  %number of nodes
Ntr_list = [100:100:1000 1250:250:2000 2500 3000];  %number of training samples
Nts = 1000;
S = 4000; %Total number of data samples as per venkitaraman2019
SNR = 5; %SNR on training data samples (target signal)
kernel_bw = 2*(20^2);

number_of_runs = 500; %MC independent experiments
NMSE_test = zeros(number_of_runs,length(Ntr_list));

%%% NETWORK DEFINITIONS %%%
param.connected = 1;
param.maxit = 200;
G = gsp_erdos_renyi(M,0.1,param); %Generating graph with edge probability = 0.2;
L = full(G.L);
PL = inv(eye(M)+L); %projection matrix for "r" into "t". argmin{...}

alpha = 1e-1;
beta = 1e-2;

C_S = iwishrnd(eye(S),S+0); %Covariance matrix from inverse wishart
sim_params = struct('M',M,'Ntr_list',Ntr_list,'Nts',Nts,'S',S,'SNR',SNR','kernel_bw',kernel_bw,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);

tic
for run = 1:number_of_runs
    fprintf('MC run %i of %i - ', run,number_of_runs);
    % DATA DEFINITIONS
    R = mvnrnd(zeros(M,S),C_S); %M rows, S columns, MxS
    % This is the transpose of venkitaraman2019
    % Columns are correlated to each other, rows are gaussian

    % generating target vectors t 
    T_total = PL*R; %#ok<*MINV>

    ones_S = ones(S,1);
    K_total = exp(  -( reshape(vecnorm( kron(R',ones_S)-kron(ones_S,R')  ,2,2).^2, S,S)  )/kernel_bw  );

    % Splitting data into training and test data
    test_indices = S-Nts+1:S;
    T_test = T_total(:,test_indices);
    
    ntr_index=0;
    for Ntr=Ntr_list

        train_indices = 1:Ntr;
        T_train = T_total(:,train_indices);
        K = K_total(train_indices,train_indices); % Training kernel matrix

        % Adding AWGN to the data
        sigma2_n = var(T_train,[],2)/(10^(0.1*SNR));
        noise = mvnrnd(zeros(Ntr,M),diag(sigma2_n))';
        T_train_n = T_train + noise;

        % Creating model with best parameters and noisy training data
        T = T_train_n';
        [U,Lambda] = eig(eye(M)+beta*L);
        [V,Sigma] = eig(K);
        lambda = diag(Lambda);
        sigma = diag(Sigma);
        P = V'*T*U;
        q = ((alpha+kron(lambda,sigma)).^(-1)).*P(:);
        Q = reshape(q,Ntr,M);
        Psi = V*Q*U';

        %%%%% TESTING %%%%%%%
        Kx = K_total(train_indices,test_indices); % "Computing" kernels with outputs
        Y_test = Psi'*Kx;
        
        ntr_index=ntr_index+1;
        NMSE_test(run,ntr_index)  = (norm(Y_test-T_test,'fro')^2/norm(T_test,'fro')^2);
    end
    toc
end

plot([0 Ntr_list], [0 10*log10(mean(NMSE_test))])