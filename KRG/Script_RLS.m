clear;
rng(3,'v5uniform');
noise_var = 0;

experiment = 'image_rec';  %klima_rec, klima_pred, brain (add _train for train)
init_get_data;

kernel_bw = 2*(5^2);
number_of_runs = 100;

D = 32;
alpha = 40;
beta = 1e-1;

% RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
Vrf = sqrt(kernel_bw^-1)*randn(input_dimension,D); % sampled vectors for RF
brf = 2*pi*rand(1,D); % random phase
Z0 = (D/2)^(-1/2)*cos(R0'*Vrf + brf)';

sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta);
% sim_params = struct('M',M,'Nts',Nts,'N',N,'noise_var',noise_var,'kernel_bw',kernel_bw,'D',D,'number_of_runs',number_of_runs,'alpha',alpha,'beta',beta,'G',G);


forget = 1;
NMSE_test = zeros(number_of_runs,Ntr);
rng(123,'v5uniform');
tic
for run=1:number_of_runs
    tic
    fprintf('MC run %i of %i - ', run,number_of_runs);

    H = zeros(D,M);
    h = H(:);
    
    S_D = alpha*eye(D*M);
    
    e = zeros(M,N);
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

    for n=1:Ntr
        z_n = Z(:,n);
        t_n = T_train(:,n);
        
        U = kron(eye(M),z_n);
        V = kron(eye(M)+beta*L, z_n');
        G = S_D*U*inv(forget*eye(M) + V*S_D*U); %#ok<*MINV>
        S_D = S_D - G*V*S_D;

        y_n = H'*z_n;
        e_n = t_n - y_n;
        
        h = h + G*(e_n - beta*L*y_n);
        H = reshape(h,D,M);

        e(:,n) = e_n;
        mse(1,n) = (1/M)*sum(e_n.^2);
        NMSE_test(run,n) = norm(H'*Z_test-T_test,'fro')^2/norm(T_test,'fro')^2;
    end  
    toc
end

% plot(10*log10(mse_average));
plot(10*log10(mean(NMSE_test,1)))
hold on