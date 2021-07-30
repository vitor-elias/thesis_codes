rng(1,'v5uniform');

M = 50;  %number of nodes
Ntr_list = [100:100:1000 1250:250:2000 2500 3000];  %number of training samples
Nts = 1000;
S = 4000; %Total number of data samples as per venkitaraman2019
SNR = 5; %SNR on training data samples (target signal)
kernel_bw = 2*(30^2);
D = 256;

number_of_runs = 500; %MC independent experiments
NMSE_test = zeros(number_of_runs,length(Ntr_list));

alpha = 1e-02;
beta = 1e-02;

%%% NETWORK DEFINITIONS %%%
param.connected = 1;
param.maxit = 200;
G = gsp_erdos_renyi(M,0.1,param); %Generating graph with edge probability = 0.2;
L = full(G.L);
PL = inv(eye(M)+L); %projection matrix for "r" into "t". argmin{...}

C_S = iwishrnd(eye(S),S+0); %Covariance matrix from inverse wishart

% RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
Vrf = sqrt(kernel_bw^-1)*randn(M,D); % sampled vectors for RF
brf = 2*pi*rand(1,D); % random phase
sim_params = struct('M',M,'Ntr_list',Ntr_list,'Nts',Nts,'S',S,'SNR',SNR','kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);
tic
for run = 1:number_of_runs
    fprintf('MC run %i of %i - ', run,number_of_runs);
    % DATA DEFINITIONS
    R = mvnrnd(zeros(M,S),C_S); %M rows, S columns, MxS
    % This is the transpose of venkitaraman2019
    % Columns are correlated to each other, rows are gaussian
    
    % Mapping R into RFF
    Z = (D/2)^(-1/2)*cos(R'*Vrf + brf);

    % generating target vectors t 
    T_total = PL*R; %#ok<*MINV>

    % Splitting data into training and test data
    test_indices = S-Nts+1:S;
    T_test = T_total(:,test_indices);
    Z_test = Z(test_indices,:);
    
    ntr_index=0;
    for Ntr=Ntr_list

        train_indices = 1:Ntr;

        T_train = T_total(:,train_indices);
        Z_train = Z(train_indices,:);

        % Adding AWGN to the data
        sigma2_n = var(T_train,[],2)/(10^(0.1*SNR));
        noise = mvnrnd(zeros(Ntr,M),diag(sigma2_n))';
        T_train_n = T_train + noise;

        % Creating model with best parameters and noisy training data
        ZT = Z_train'*T_train_n';
        
    %     B = kron(eye(M),(Z_train'*Z_train + alpha*eye(D)));
    %     C = kron(beta*L,Z_train'*Z_train);
    %     Psi = reshape(inv(B+C)*ZT(:),D,M);

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
plot([0 Ntr_list], [0 10*log10(mean(NMSE_test))])
% plot([0 Ntr_list], [0 10*log10((NMSE_test))])
hold on