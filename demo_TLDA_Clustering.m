
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'new_ORL'};
k1 = 4;
k2 = 4;
%=============================================
for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', name);
    fprintf('Loading   ...');
    load( name );% loading dataset
    fprintf('Done\n');
    nclass = length(find(gnd==1));
    % =====================================================================
    type_method = 'TLDA'; % as type = 'NTD_LE' & options.alpha = 0, that is NTD algorithm
    tensor_shape    = [32 32];% the dimension for dataset dependent reshaped tensors
    lowerdims = [6 6];
    % =====================================================================
    runtimes = 4; % runing 10 times
    cross_num = 5;
    % =====================================================================
    for rn = 1 : runtimes
        indx = zeros(nclass*rn*10, 1);
        n = 10*rn +1;
        indx(:,1) = find(gnd<n);
        indx = indx(:);
        X = fea( indx, :);                         %===================%
        Y = gnd( indx, 1);                     %      选择训练集    %
        t1 = clock;% count time
        for cros = 1 : cross_num
            train = X';                         %===================%
            train_label = Y;                     %      选择训练集    %
            nsamples = size(X, 1);
            nClasses = length(unique(Y));
            % =============================================================
            [Us, ela] = TDLA_func(train,train_label',tensor_shape,lowerdims,k1,k2);
            % ============================== train  =======================
            trainCore = CoreTensor(train, Us, tensor_shape);
            V = reshape(trainCore, [prod(lowerdims) nsamples]);
            label = litekmeans(V.data', nClasses,'Replicates', 20);
            label = bestMap(train_label, label);
            results.AC(rn, cros) = length(find(train_label == label))/length(label);
            results.NMI(rn, cros) = MutualInfo(train_label,label);
            iinf = cros + (rn - 1)*cross_num;
            results.rse{iinf} = ela;
            clear V label;   
        end
    end
    results.method = type_method;
    save(['Results\', 'resC_', type_method,'', name ],  'results');
    clear results;
end

