for mu = 0.5 %Iterates over different step sizes
    tic
    rng(4,'v5uniform');
    % GENERAL PARAMETERS
    T = 4000; %Total time
    M = 4; %Filter order
    N = 20; %Number of nodes

    number_of_runs = 1000; %Monte Carlo Runs
    MSE_average_cc = zeros(1,T);
    MSE_average_rf = zeros(1,T);
    MSE_average_lms = zeros(1,T);
    %step sizes
    mu_cc = 0.25;
    mu_rf = 0.75;

    %kernel setup
    kernel_bw = 2;
    kernel = @(z,zin) exp(-(norm(z-zin)^2)/kernel_bw);

    % PARAMS CC
    cc_param = 0.5; %0.1 for 32, 0.2 or 0.3 for 64, 0.4 for 128
    dict_target = 256; %Targeted dictionary size

    % PARAMS RF
    D = dict_target; %Making RFF-space dimension equal to the dictionary size

    % NETWORK
    g=load('G');
    G=g.G;
        A = full(G.A);
        L = full(G.L);
        c=eye(N) + A;
        count=sum(c');

    CC = zeros(N,N);
    for i=1:N
        for j=1:N
            if(i~=j)
                if c(i,j)~=0
                    Cc(i,j)=1/(max(count(i),count(j)));
                else
                    Cc(i,j)=0;
                end
            else
                Cc(i,j)=0;
            end
        end
        Cc(i,i)= 1 - sum(Cc(i,:));
    end
    C_nl = Cc;


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

    % CC DICTIONARY TRAINING
    Xtrain = mvnrnd(zeros(N,1),Rx,5*T+M+1)';
    for t=1:length(Xtrain)
        for m = 0:min(t-1,M-1)  % ICASSP Z
            Ztrain(:,m+1,t) = (S^m)*Xtrain(:,t-m);
        end
    end
    dict = Ztrain(1,:,M);
    while length(dict)~=dict_target
        Xtrain = mvnrnd(zeros(N,1),Rx,5*T+M+1)';

        for t=1:length(Xtrain)
            for m = 0:min(t-1,M-1)  % ICASSP Z
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
    Vp = sqrt(kernel_bw^-1)*randn(M,D); % sampled vectors for RF
    b = 2*pi*rand(D,1); % random phase

    % Kernel LMS MC iterations
    for run=1:number_of_runs % MC iteration
        fprintf('run: %i - ', run);
        
        % INPUT X AND NOISE V    
        X = mvnrnd(zeros(N,1),Rx,T)';
        V = mvnrnd(zeros(N,1),Rv,T)';

        % Definition of Z
        % Non-linear filter 
        Y = zeros(N,T);
        Z = zeros(N,M,T);

        % Coherence-check variables
        alpha = mat2cell(zeros(dict_target,20),dict_target,ones(1,20));
        beta = alpha;
        kvec = alpha;

        % RF variables
        theta = zeros(D,N); %RF coefficients
        psi = zeros(D,N); %Auxiliary RF coefficients (combine terms)

        % LMS variables
        psi_hat = zeros(M,N);
        h_dist = zeros(M,N);

        % error measures
        e_cc = zeros(N,T);
        mse_cc = zeros(N,T);

        e_rf = zeros(N,T);
        mse_rf = zeros(N,T);

        e_lms = zeros(N,T);
        mse_lms = zeros(N,T);

        % iterations
        for t=1:T

            %INPUT VECTOR AND OUTPUT FOR THAT ITERATION
            for m = 0:min(t-1,M-1)
                Z(:,m+1,t) = (S^m)*X(:,t-m);
            end
            Y(:,t) = sqrt(Z(:,1,t).^2 + sin(Z(:,4,t)*pi).^2) + (0.8 - 0.5*exp(-Z(:,2,t).^2)).*Z(:,3,t) + V(:,t);


            % LMS
            for k = 1:N %ADAPT
                %ADAPT CC PT1
                kvec{k} = exp(-(vecnorm(dict-Z(k,:,t),2,2).^2)/kernel_bw);
                e_cc(k,t) = Y(k,t) - alpha{k}'*kvec{k};
                mse_cc(k,t)= e_cc(k,t)^2;

                %ADAPT RF
                zk = Z(k,:,t)'; %input vector at node k;
                rf_k = (D/2)^(-1/2)*cos(Vp'*zk + b); 

                e_rf(k,t) = Y(k,t) - theta(:,k)'*rf_k;  % current error   
                mse_rf(k,t)= e_rf(k,t)^2;
                psi(:,k) = theta(:,k) + mu_rf*e_rf(k,t)*rf_k;  %updating theta

                %ADAPT LMS
                psi_hat(:,k) = h_dist(:,k) + mu*zk*(Y(k,t) - zk'*h_dist(:,k));
                e_lms(k,t) = Y(k,t) - zk'*h_dist(:,k);
                mse_lms(k,t)= e_lms(k,t)^2;
            end

            for k = 1:N %ADAPT CC PT2
                aux_beta = alpha{k};
                N_k = find(C_nl(k,:));
                for l = N_k
                    e_kl = Y(l,t) - alpha{k}'*kvec{l};
                    aux_beta = aux_beta + mu_cc*C_nl(k,l)*(e_kl)*kvec{l};
                end
                beta{k} = aux_beta;
            end

            % COMBINE %
            %COMBINE CC
            for k = 1:N
                alpha{k} = cell2mat( beta )*Cc(k,:)';
            end
            %COMBINE RF
            theta  = psi*Cc;

            %COMBINE LMS
            h_dist = psi_hat*Cc;

        end
        MSE_cc = (1/N)*sum(mse_cc);
        MSE_average_cc = MSE_average_cc + MSE_cc;

        MSE_rf = (1/N)*sum(mse_rf);
        MSE_average_rf = MSE_average_rf + MSE_rf;

        MSE_lms = (1/N)* sum(mse_lms);
        MSE_average_lms = MSE_average_lms + MSE_lms;

        toc    
    end

    MSE_final_average_rf = MSE_average_rf/number_of_runs;
    MSE_final_average_cc = MSE_average_cc/number_of_runs;
    MSE_final_average_lms = MSE_average_lms/number_of_runs;
    plot(10*log10(MSE_final_average_cc),'r');
    hold on;
    plot(10*log10(MSE_final_average_rf),'g');
    plot(mag2db(MSE_final_average_lms),'r');
end