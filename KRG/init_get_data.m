switch experiment 
    
    case 'klima_rec'
        data_file = './Data/Klima_data_2019.mat';
        data = load(data_file);
        R0 = data.humid;
        T0 = data.temp;
        pos = data.pos;
        station_indexes = data.station_indexes;
        R0 = R0(station_indexes,:);
        T0 = T0(station_indexes,:);
        pos = pos(station_indexes,:);
        graph_param.k = 5;
        G = gsp_nn_graph(pos,graph_param);
        L = full(G.L);
        train_ratio = 0.7;
        
    case 'klima_pred'
        data_file = './Data/Klima_data_2019.mat';
        data = load(data_file);
        R0 = data.temp;
        T0 = data.temp;
        pos = data.pos;
        station_indexes = data.station_indexes;
        R0 = R0(station_indexes,1:end-4);
        T0 = T0(station_indexes,1+4:end);
        pos = pos(station_indexes,:);
        graph_param.k = 5;
        G = gsp_nn_graph(pos,graph_param);
        L = full(G.L);
        train_ratio = 0.7;
        
    case 'brain'
        data_file = './Data/Brain_data';
        data = load(data_file);
        R0 = data.Data;
        T0 = data.Data;
        R0 = R0(1,:);
        T0 = T0(11:end,:);
        train_ratio = 146/292;
        L = data.L;
        noise_var = 0.1;
        
    case 'brain_train'
        data_file = './Data/Brain_data';
        data = load(data_file);
        R0 = data.Data;
        T0 = data.Data;
        R0 = R0(1:10,data.train_indices);
        train_nodes = 11:30;
        T0 = T0(train_nodes,data.train_indices);
        train_ratio = 0.5;
        L = data.L;
        L = L(train_nodes,train_nodes);
        
    case 'image_rec'
        data_file = './Data/ImageRec_data';
        data = load(data_file);
        R0 = data.badblocks;
%         R0 = R0(:,1:2*14400);
        T0 = data.realblocks;
%         T0 = T0(:,1:2*14400);
        G = data.G;
        L = G.L;
        train_ratio = 6/8;
        noise_var = 0;
        
    case 'image_rec_train'
        data_file = './Data/ImageRec_data';
        data = load(data_file);
        R0 = data.badblocks;
        R0 = R0(:,60001:63000);
        T0 = data.realblocks;
        T0 = T0(:,60001:63000);
        G = data.G;
        L = G.L;
        train_ratio = 1/5;
        noise_var = 0;
        
    case 'frame_pred'
        data_file = './Data/FramePred_data';
        data = load(data_file);
        nframes = 10;
        R0 = data.current;
        R0 = R0(:,9*14400+1:(9+nframes)*14400);
        T0 = data.next;
        T0 = T0(:,9*14400+1:(9+nframes)*14400);
        G = data.G;
        L = G.L;
        train_ratio = 8/10;
        noise_var = 0;

    case 'frame_pred_train'
        data_file = './Data/FramePred_data';
        data = load(data_file);
        nframes = 2;
        R0 = data.current;
        R0 = R0(:,1:nframes*14400);
        T0 = data.next;
        T0 = T0(:,1:nframes*14400);
        G = data.G;
        L = G.L;
        train_ratio = 1/2;
        noise_var = 0;
        
    case 'frameBMC_pred'
        data_file = './Data/FramePredBMC_data';
        data = load(data_file);
        nframes = 9;
        R0 = data.bmcref;
        R0 = R0(:,1:nframes*14400);
        T0 = data.target;
        T0 = T0(:,1:nframes*14400);
        G = data.G;
        L = G.L;
        train_ratio = 8/9;
        noise_var = 0;
        
    case 'filtered'
        data_file = './Data/Filtered_data2';
        data = load(data_file);
        indices_A = data.indices_A;
        indices_B = data.indices_B;
        nsamples = 20e6;
        R0 = zeros(length(indices_A),nsamples);
        T0 = zeros(length(indices_B),nsamples); % choosing B as target so get L_B;
        G = data.G;
        L = full(G.L);
        train_ratio = (nsamples - 1000)/nsamples;
        
        V = data.V;
        filter_mask = zeros(length(indices_A)+length(indices_B),1); filter_mask(1:4) = 1;
        Hfilter = V*diag(filter_mask)*V';
end

clear data;
[M,Ndata] = size(T0);
N = floor(train_ratio*Ndata);
Ntr_list = 1:N;
Ntr = N;
Nts = Ndata-N;
ones_Nts = ones(Nts,1);
input_dimension = size(R0,1);
