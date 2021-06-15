clear;
rng(3,'v5uniform');
noise_var = 0.0;

experiment = 'klima_pred';  %klima_rec, klima_pred, brain (add _train for train)
init_get_data;

kernel_bw = 2*(50^2);
number_of_runs = 10000;

D = 128;
alpha = 1e-14;
beta = 1e-2;

mu = 0.005;
Nb = 50;

% mu = 0.0025;
% Nb = 30;

sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'mu',mu,'Nb',Nb);
% sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);

% RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
Vrf = sqrt(kernel_bw^-1)*randn(input_dimension,D); % sampled vectors for RF
brf = 2*pi*rand(1,D); % random phase
Z0 = (D/2)^(-1/2)*cos(R0'*Vrf + brf)';
%
NMSE_test = zeros(1,Ntr);
mse_average = zeros(1,Ntr);

rng(123,'v5uniform');
tic
for run=1:number_of_runs
    tic
    fprintf('MC run %i of %i - ', run,number_of_runs);

    Hi = zeros(D,M);
    mse = zeros(1,Ntr);    
    NMSE = zeros(1,Ntr);
    
    permutindex = randperm(Ndata);
    Z = Z0(:,permutindex);
    T = T0(:,permutindex);
    
    train_indices = 1:Ntr;        
    Z_train = Z(:,train_indices);
    T_train = T(:,train_indices);
    
    Z_test = Z(:,end-Nts+1:end);
    T_test = T(:,end-Nts+1:end);

    % Adding AWGN to the data
    T_train = T_train + sqrt(noise_var)*randn(M,Ntr);

    for n = 1:Ntr
        a = n-Nb+1; %batch start
        if a<1,a=1;end
        Zi = Z(:,a:n)';
        Ti = T_train(:,a:n)';
        
        z_n = Z(:,n);
        t_n = T_train(:,n);

        y_n = Hi'*z_n;
        e_n = t_n - y_n;

        Hi = Hi+mu*(Zi'*(Ti-Zi*Hi - beta*Zi*Hi*L)-alpha*Hi);                

        mse(1,n) = (norm(e_n)^2)/(norm(t_n)^2);
        NMSE(1,n) = norm(Hi'*Z_test-T_test,'fro')^2/norm(T_test,'fro')^2;
    end
    mse_average = mse_average + (mse./number_of_runs);
    NMSE_test = NMSE_test + (NMSE./number_of_runs); 
    
    toc
end
plot(10*log10(mse_average));
% plot(10*log10(NMSE_test))
hold on