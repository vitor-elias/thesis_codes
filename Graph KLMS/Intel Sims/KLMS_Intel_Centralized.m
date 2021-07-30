clear all; clc;
tic

M = 5; %Filter order


kernel_bw = 20;
kernel = @(z,zin) exp(-(norm(z-zin)^2)/kernel_bw);

% NETWORK
intel_dataset = load('intel_data_and_structure_nn4.mat');

G = intel_dataset.G; %graph
W = intel_dataset.W; %shift/weight matrix (gaussian from distances)
S = W;
[VS,DS] = eig(S);
S = VS*( DS./max(diag(abs(DS))))*VS';
S(S<10^-5) = 0;

N = G.N;

A = full(G.A);
L = full(G.L);


% nv=load('nv');
sigma_v = 0.1+rand(N,1)/20;

% Rx = diag(sigma_x);
Rv = diag(sigma_v);

data = load('intel_data_5min_march.mat');
X = data.Xtemp;
Y = data.Xhumid;
time_stamps = data.Time_stamps;    
T = length(X);

number_of_runs = 1;
iter=0;
for mu = 0.07
    rng(4,'v5uniform');
    error_average = zeros(1,T); %Stores MC-average of error per time instant
    MSE_average_cc = zeros(1,T);
    MSE_average_lms = zeros(1,T);
    MSE_average_cent = zeros(1,T);
    MSE_average_rff_d1 = zeros(1,T);
    % MSE_average_rff_d2 = zeros(1,T);
    mu_rff = 0.03;
    mu_cc = 0.03;



    % PARAMS CC
    cc_param = 0.4; %0.1 for 32, 0.2 or 0.3 for 64, 0.4 for 128
    dict_target = 128; %Targeted dictionary size

    %PARAM RFF
    D1 = dict_target; %Making RFF-space dimension equal to the dictionary size
    % D2 = 128;


    % CC DICTIONARY TRAINING
    Xtrain = X;
    dict = zeros(1,M);
    while length(dict)~=dict_target
        for t=1:length(Xtrain)
            for m = 0:min(t-1,M-1)  % ICASSP Z
                Ztrain(:,m+1,t) = (S^m)*Xtrain(:,t-m);
            end
        end
        
        for t=5:length(Xtrain)    
            for k = 1:N
                zk = Ztrain(k,:,t);
                kvec = exp(-(vecnorm(dict-zk,2,2).^2)/kernel_bw);
                if max(kvec) <= cc_param
                    dict = [dict;zk];
    %                 fprintf('size %i/n',length(dict));
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

    % Vp2 = sqrt(kernel_bw^-1)*randn(M,D2); % sampled vectors for RF
    % b2 = 2*pi*rand(D2,1); % random phase
    % B2 = repmat(b2',N,1);

    tic
    for run=1:number_of_runs % MC iteration
        fprintf('run: %i - ',run);

        V = mvnrnd(zeros(N,1),Rv,T)';

        % Definition of Z (instantaneous or temporal)
        % Non-linear filter (non-linear function: norm(.))
        Z = zeros(N,M,T);


        %Coherence-check variable;
        alpha = zeros(dict_target,1);

        %RFF variables
        h1 = zeros(D1,1);
    %     h2 = zeros(D2,1);

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

    %     e_rff2 = zeros(N,T);
    %     mse_rff2 = zeros(N,T);

        ones_N = ones(N,1);
        for t=1:T

            %INPUT VECTOR AND OUTPUT FOR THAT ITERATION
            for m = 0:min(t-1,M-1);  % ICASSP Z
                Z(:,m+1,t) = (S^m)*X(:,t-m);
            end

            Zt = Z(:,:,t);

            % CENTRALIZED GRAPH KLMS 
    %         f_Zt = zeros(N,1);        
    %         for tau = 1:t-1
    %             Ztau = Z(:,:,tau);
    %             f_Zt = f_Zt + exp(  -( reshape(vecnorm( kron(ones_N,Zt)-kron(Ztau,ones_N)  ,2,2).^2, N,N)  )/kernel_bw  )*mu*e_cent(:,tau);
    %         end
    %         
    %         e_cent(:,t) = Y(:,t) - f_Zt;
    %         mse_cent(:,t) = e_cent(:,t).^2;

            % CENTRALIZED RFF1
            Rt1 = (D1/2)^(-1/2)*cos(Zt*Vp1 + B1);
            e_rff1(:,t) = Y(:,t) - Rt1*h1;
%             mse_rff1(:,t) = (e_rff1(:,t).^2)./(Y(:,t)'*Y(:,t));
%             mse_rff1(:,t) = (e_rff1(:,t).^2)./(Y(:,t).^2); %<<<
            mse_rff1(:,t) = (e_rff1(:,t).^2);

            h1 = h1 + mu_rff*Rt1'*e_rff1(:,t);

    %         % CENTRALIZED RFF2
    %         Rt2 = (D2/2)^(-1/2)*cos(Zt*Vp2 + B2);
    %         e_rff2(:,t) = Y(:,t) - Rt2*h2;
    %         mse_rff2(:,t) = e_rff2(:,t).^2;
    %         
    %         h2 = h2 + mu*Rt2'*e_rff2(:,t);

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
%             mse_cc(:,t) = (e_cc(:,t).^2)./(Y(:,t)'*Y(:,t));
%             mse_cc(:,t) = (e_cc(:,t).^2)./(Y(:,t).^2); %<<<
            mse_cc(:,t) = (e_cc(:,t).^2);
            alpha = alpha + mu_cc*Kbf'*e_cc(:,t);

        end

        MSE_cc = (1/N)*sum(mse_cc);
        MSE_average_cc = MSE_average_cc + MSE_cc;

        MSE_lms = (1/N)*sum(mse_lms);
        MSE_average_lms = MSE_average_lms + MSE_lms;

    %     MSE_cent = (1/N)*sum(mse_cent);
    %     MSE_average_cent = MSE_average_cent + MSE_cent;

        MSE_rff1 = (1/N)*sum(mse_rff1);
        MSE_average_rff_d1 = MSE_average_rff_d1 + MSE_rff1;

    %     MSE_rff2 = (1/N)*sum(mse_rff2);
    %     MSE_average_rff_d2 = MSE_average_rff_d2 + MSE_rff2;

        toc
    end

    MSE_average_lms_final = MSE_average_lms/number_of_runs;
    % MSE_average_cent_final = MSE_average_cent/number_of_runs;
    MSE_average_rff_final1 = MSE_average_rff_d1/number_of_runs;
    MSE_average_cc_final = MSE_average_cc/number_of_runs;
    % MSE_average_rff_final2 = MSE_average_rff_d2/number_of_runs;
    
%     %NMSE PLOTS
    plot(10*log10(MSE_average_cc_final),'r');
    hold on;
    plot(10*log10(MSE_average_rff_final1),'g');
    
%     subplot(2,1,1);
%     k1 = 40;
%     plot(Y(k1,:),'b', 'linewidth',2);
%     hold on;
%     plot(Y(k1,:)-e_cc(k1,:),'r');
%     plot(Y(k1,:)-e_rff1(k1,:),'g');
%     axis tight;
% %     xlabel('Iteration index (\it{n})');
%     ylabel('Humidity (%)');
%     legend('Original','CC-based GKLMS','RFF-based GKLMS');
%     xlim([0 1400]);
    
%     plot(mag2db(MSE_average_cc_final),'r');
%     hold on
%     plot(mag2db(MSE_average_rff_final1),'g');
%     hold on;
%     plot(mag2db(MSE_average_lms_final),'r');
%     plot(mag2db(MSE_average_cc_final),'g');
    % plot(mag2db(MSE_average_cent_final),'k');
%     xlabel('Iteration','FontName','Times New Roman','FontSize',14);
%     ylabel('MSE [dB]','FontName','Times New Roman','FontSize',14);
    % 
    % fig_legend = legend('RFF KLMS $D=32$','LMS','CC');
    % fig_legend.FontSize = 14;
    % fig_legend.Interpreter = 'latex';
    
%     iter = iter+1;
%     Legend{iter} = strcat('RFF ', num2str(mu));
%     iter = iter+1;
%     Legend{iter} = strcat('CC ', num2str(mu));
%     fig_legend = legend(Legend);

%     save(['MSE_CENT_D' num2str(D1) 'mu' num2str(mu) '.mat'],'MSE_average_rff_final1','MSE_average_cc_final','mu_rff','mu_cc','D1','cc_param','number_of_runs','T')
    % save(['MSE_ALL_CENT_D1_' num2str(D1) 'mu00' num2str(mu*100)],'MSE_average_lms_final','MSE_average_cent_final','MSE_average_rff_final1','MSE_average_cc_final','mu','D1','cc_param','number_of_runs','T')
    % save(['MSE_ALL_CENT_D1_' num2str(D1) 'D2_' num2str(D2) 'mu0' num2str(mu*10)],'MSE_average_lms_final','MSE_average_cent_final','MSE_average_rff_final1','MSE_average_rff_final2','mu','D1','D2','number_of_runs','T')
end