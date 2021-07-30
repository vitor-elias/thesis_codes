clear
rng(1,'v5uniform');

M = 50;  %number of nodes
Ntr = 3000;  %number of training samples
Nts = 1000;
S = 4000; %Total number of data samples as per venkitaraman2019
SNR = 5; %SNR on training data samples (target signal)
kernel_bw = 2*(30^2);
D = 32;

number_of_runs = 200; %MC independent experiments
NMSE_train = zeros(1,number_of_runs);
NMSE_test = zeros(1,number_of_runs);
norm_test = zeros(1,number_of_runs);

%%% NETWORK DEFINITIONS %%%
param.connected = 1;
param.maxit = 200;
G = gsp_erdos_renyi(M,0.1,param); %Generating graph with edge probability = 0.2;
L = full(G.L);
Q = inv(eye(M)+L); %projection matrix for "r" into "t". argmin{...}

C_S = iwishrnd(eye(S),S+0); %Covariance matrix from inverse wishart

% RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
Vrf = sqrt(kernel_bw^-1)*randn(M,D); % sampled vectors for RF
brf = 2*pi*rand(1,D); % random phase

mse_average = zeros(1,Ntr);
NMSE_average = zeros(1,Ntr);

forget = 0.85;
alpha = 1e-2;
beta = 1e-2;

sim_params = struct('M',M,'Ntr',Ntr,'Nts',Nts,'S',S,'SNR',SNR','kernel_bw',kernel_bw,...
'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);

for run=1:number_of_runs
    tic
    fprintf('MC run %i of %i - ', run,number_of_runs);

    H = zeros(D,M);
    h = H(:);
    
    S_D = 50*eye(D*M);
    
    e = zeros(M,S);
    mse = zeros(1,Ntr);
    NMSE = zeros(1,Ntr);
    
    % DATA DEFINITIONS
    R = mvnrnd(zeros(M,S),C_S); %M rows, S columns, MxS
    % This is the transpose of venkitaraman2019
    % Columns are correlated to each other, rows are gaussian
    
    train_indices = 1:Ntr;
    test_indices = S-Nts+1:S;
    
    % Mapping R into RFF
    Z = (D/2)^(-1/2)*cos(R'*Vrf + brf)';
    Z_train = Z(:,train_indices);
    Z_test = Z(:,test_indices);

    % generating target vectors t 
    T = Q*R; %#ok<*MINV>
    T_train = T(:,train_indices);
    T_test = T(:,test_indices);

    % Adding AWGN to the data
    sigma2_n = var(T,[],2)/(10^(0.1*SNR));
    noise = mvnrnd(zeros(Ntr,M),diag(sigma2_n))';
    T_train = T_train + noise;

    for n=1:Ntr
        z_n = Z(:,n);
        t_n = T_train(:,n);
        
        U = kron(eye(M),z_n);
        V = kron(eye(M)+beta*L, z_n');
        G = S_D*U*inv(forget*eye(M) + V*S_D*U);
        S_D = S_D - G*V*S_D;

        y_n = H'*z_n;
        e_n = t_n - y_n;
        
        h = h + G*(e_n - beta*L*y_n);
        H = reshape(h,D,M);

%         H = H + mu*(z_n*e_n' - alpha*H - beta*(z_n*z_n')*H*L);

        e(:,n) = e_n;
        mse(1,n) = (1/M)*sum(e_n.^2);
        NMSE(1,n) = norm(H'*Z_test-T_test,'fro')^2/norm(T_test,'fro')^2;
    end
    mse_average = mse_average + (mse./number_of_runs);
    NMSE_average = NMSE_average + (NMSE./number_of_runs);

%     %%%%% TESTING %%%%%%%
%     Y_test = H'*Z_test;
%     NMSE_test(run) = 10*log10(norm(Y_test-T_test,'fro')^2/norm(T_test,'fro')^2);
    
    toc
end

% plot(10*log10(mse_average));
plot(10*log10(NMSE_average))