tic
T = 4000; %Total time
M = 4; %Filter order
N = 20; %Number of nodes

number_of_runs = 1000;
iter=0;
for mu = 0.01 %can iterate for different step-sizes
    rng(4,'v5uniform'); %setting a random seed
    error_average = zeros(1,T); %Stores MC-average of error per time instant
    MSE_average_cc = zeros(1,T);
    MSE_average_lms = zeros(1,T);
    MSE_average_cent = zeros(1,T);
    MSE_average_rff_d1 = zeros(1,T);
    %step sizes
    mu_rff = mu;
    mu_cc = mu;

    %kernel setup
    kernel_bw = 2;
    kernel = @(z,zin) exp(-(norm(z-zin)^2)/kernel_bw);

    % PARAMS CC
    cc_param = 0.5; %0.1 for 32, 0.2 or 0.3 for 64, 0.4 for 128
    dict_target = 256; %Targeted dictionary size

    %PARAM RFF
    D1 = dict_target; %Making RFF-space dimension equal to the dictionary size

    % NETWORK
    g=load('G');
    G=g.G;
        A = full(G.A);
        L = full(G.L);
        c=eye(N) + A;
        count=sum(c');

    W = rand(N,N);
    W = (W+W')/2;
    S = W.*A;
    [VS,DS] = eig(S);
    S = VS*( DS./max(diag(abs(DS))))*VS';
    S(S<10^-5) = 0;

    % SIGNAL PARAMS    
    sv=load('sv');
    sigma_x = sv.sigma_x;

    nv=load('nv');
    sigma_v = nv.sigma_v;

    Rx = diag(sigma_x);
    Rv = diag(sigma_v);

    % CC Dictionary training

    Xtrain = mvnrnd(zeros(N,1),Rx,5*T+M+1)';
    for t=1:length(Xtrain)
        for m = 0:min(t-1,M-1) 
            Ztrain(:,m+1,t) = (S^m)*Xtrain(:,t-m);
        end
    end
    dict = Ztrain(1,:,M);
    while length(dict)~=dict_target
        Xtrain = mvnrnd(zeros(N,1),Rx,5*T+M+1)';

        for t=1:length(Xtrain)
            for m = 0:min(t-1,M-1) 
                Ztrain(:,m+1,t) = (S^m)*Xtrain(:,t-m);
            end
        end

        for t = M+1:length(Ztrain)
            for k = 1:N
                zk = Ztrain(k,:,t);
                kvec = exp(-(vecnorm(dict-zk,2,2).^2)/kernel_bw);
                if max(kvec) <= cc_param
                    dict = [dict;zk];
                end
                if length(dict) == dict_target, break; end
            end
            if length(dict) == dict_target, break; end
        end
    end

    % RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
    Vp1 = sqrt(kernel_bw^-1)*randn(M,D1); % sampled vectors for RF
    b1 = 2*pi*rand(D1,1); % random phase
    B1 = repmat(b1',N,1);

    tic
    for run=1:number_of_runs % MC iteration
        fprintf('run: %i - ',run);

        X = mvnrnd(zeros(N,1),Rx,T)';
        V = mvnrnd(zeros(N,1),Rv,T)';

        %Setting variables
        Y = zeros(N,T);
        Z = zeros(N,M,T);

        %Coherence-check variable;
        alpha = zeros(dict_target,1);

        %RFF variables
        h1 = zeros(D1,1);

        %LMS variables
        h_cent = zeros(M,1);

        % error measures
        e_cc = zeros(N,T);
        mse_cc = zeros(N,T);

        e_lms = zeros(N,T);
        mse_lms = zeros(N,T);

        e_cent = zeros(N,T);
        mse_cent = zeros(N,T); 

        e_rff1 = zeros(N,T);
        mse_rff1 = zeros(N,T);

        ones_N = ones(N,1);
        for t=1:T

            %INPUT VECTOR AND OUTPUT FOR THAT ITERATION
            % Definition of Z (instantaneous or temporal)
            for m = 0:min(t-1,M-1)
                Z(:,m+1,t) = (S^m)*X(:,t-m);
            end
            Y(:,t) = sqrt(Z(:,1,t).^2 + sin(Z(:,4,t)*pi).^2) + (0.8 - 0.5*exp(-Z(:,2,t).^2)).*Z(:,3,t) + V(:,t);

            Zt = Z(:,:,t);

            % CENTRALIZED GRAPH KLMS 
            f_Zt = zeros(N,1);        
            for tau = 1:t-1
                Ztau = Z(:,:,tau);
                f_Zt = f_Zt + exp(  -( reshape(vecnorm( kron(ones_N,Zt)-kron(Ztau,ones_N)  ,2,2).^2, N,N)  )/kernel_bw  )*mu*e_cent(:,tau);
            end
            
            e_cent(:,t) = Y(:,t) - f_Zt;
            mse_cent(:,t) = e_cent(:,t).^2;

            % CENTRALIZED RFF1
            Rt1 = (D1/2)^(-1/2)*cos(Zt*Vp1 + B1);
            e_rff1(:,t) = Y(:,t) - Rt1*h1;
            mse_rff1(:,t) = e_rff1(:,t).^2;

            h1 = h1 + mu_rff*Rt1'*e_rff1(:,t);

            % LINEAR LMS
            e_lms(:,t) = Y(:,t) - Z(:,:,t)*h_cent;
            mse_lms(:,t) = e_lms(:,t).^2;
            h_cent = h_cent + 0.5*mu*Z(:,:,t)'*e_lms(:,t);

            % Coherence-check KLMS
            Kbf = zeros(N,dict_target);
            for k=1:N
                Kbf(k,:) = exp(-(vecnorm(dict'-Z(k,:,t)',2,1).^2)/kernel_bw);
            end
            e_cc(:,t) = Y(:,t) - Kbf*alpha;
            mse_cc(:,t) = e_cc(:,t).^2;
            alpha = alpha + mu_cc*Kbf'*e_cc(:,t);

        end

        MSE_cc = (1/N)*sum(mse_cc);
        MSE_average_cc = MSE_average_cc + MSE_cc;

        MSE_lms = (1/N)*sum(mse_lms);
        MSE_average_lms = MSE_average_lms + MSE_lms;

        MSE_cent = (1/N)*sum(mse_cent);
        MSE_average_cent = MSE_average_cent + MSE_cent;

        MSE_rff1 = (1/N)*sum(mse_rff1);
        MSE_average_rff_d1 = MSE_average_rff_d1 + MSE_rff1;

        toc
    end

    MSE_average_lms_final = MSE_average_lms/number_of_runs;
    MSE_average_cent_final = MSE_average_cent/number_of_runs;
    MSE_average_rff_final1 = MSE_average_rff_d1/number_of_runs;
    MSE_average_cc_final = MSE_average_cc/number_of_runs;

    plot(10*log10(MSE_average_cc_final),'r');
    hold on;
    plot(10*log10(MSE_average_rff_final1),'g');
    plot(mag2db(MSE_average_lms_final),'r');
    plot(mag2db(MSE_average_cent_final),'k');
    xlabel('Iteration','FontName','Times New Roman','FontSize',14);
    ylabel('MSE [dB]','FontName','Times New Roman','FontSize',14);

end
