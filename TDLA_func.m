function [Us, ela] = TDLA_func(train,train_label,tensor_shape,lowerdims,k1,k2)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
ela = cputime;
[~, N] = size(train);
train_num = N;
I = tensor_shape;
J = lowerdims; %%%%%%%%%%%%%%%%%%%%%NOT SURE
X = reshape( train,[I(1),I(2),N]);
train_set = tensor(X,[I(1),I(2),N]);
%******************************选择矩阵Si********************                           % k1个同类的和k2个不同类的，以及中心像素Xi本身组合成 Patch(Xi)
all_S_matrix = cell(1,train_num);             % 存放所有训练样本的选择矩阵的数据结构   
for i = 1:train_num                           % Si矩阵对每个Xi样本而言  N×(n+1) 每个样本i，N行 n+1 列  Si(a,b) 表示第b列第a行，
                                                % 如果值为1，表明N个样本中第a个样本为样本Xi的Patch中的第a个像素
    sample_E_dis = zeros(1,train_num);        % 样本train_set{1,i} 到所有样本的欧式距离存放在 sample_E_dis中，排序找最近的n1和n2 构成Si矩阵，放到all_S_matrix中
    select_matrix = zeros(train_num,k1+k2+1);
    for j = 1:train_num
        re = train_set(:,:,i) - train_set(:,:,j);
        re = tenmat(re,1);
        re = re.data;
        re = re.^2;
        re = sum(re(:));
        sample_E_dis(1,j) = sqrt(re);           % sample_E_dis 表示当前样本与其他所有样本（包括自己）的欧式距离
    end
    % end for 求样本Xi到其他所有样本的欧式距离sample_E_dis(1,sam_num)
    % [sample_E_dis,~] = mapminmax(sample_E_dis,0,1);
    [~,pos_E_dis] = sort(sample_E_dis,'ascend');
    
    select_matrix(pos_E_dis(1,1),1)=1;          % 选择矩阵第一列肯定是欧式距离为0的中心像素
    
    ind = 2;
    for k=2:(k1+1)                              % [2,...,n1+1]是n1个与样本同类的近距离点
        while train_label(1,pos_E_dis(1,ind)) ~= train_label(1,pos_E_dis(1,1))  % 不同类则直接跳过
            ind = ind + 1;
        end
        select_matrix(pos_E_dis(1,ind),k) = 1;   % select_matrix(i,j) = 1 表示第Xi的patch中 第j个位置像素对应在总样本sample_cell的第i个位置  
        ind = ind + 1;
    end
    % end for 求n1 个同类且最近的样本的编号，并把Xi的选择矩阵Si中对应列的位置标为1
    
    ind = 2;
    for k = (k1+2):(k1+k2+1)                     % [n1+2,...,n1+n2+1]是n2个与样本不同类的近距离点
        while train_label(1,pos_E_dis(1,ind)) == train_label(1,pos_E_dis(1,1))  % 从欧式距离最小找起，同类的则跳过
           ind = ind + 1;
        end
        select_matrix(pos_E_dis(1,ind),k) = 1;
        ind = ind + 1;
    end
    %end for 求n2 个不同类且最近的样本的编号，并把Xi的选择矩阵Si中对应列的位置标为1
    
    all_S_matrix{1,i} = select_matrix;   %把Xi的选择矩阵放到总的选择矩阵中去
end


%*********先求论文中的Q矩阵*********
s_weight = 1;                       % patch 中同类样本的权重 = 论文中为1     
dif_weight = -4;                    % patch 中不同类样本中的权重 = 论文中的a   β= [1,1,1,1,1,a,a,a,a,a]
Belta = [s_weight*ones(k1,1);dif_weight*ones(k2,1)];
Belta = Belta;
sum_Belta = sum(Belta);
left_vector = [sum_Belta;-Belta];   % Q的左边部分
right_mat = [-Belta';diag(Belta)];  % Q的右边部分
Q_matrix = [left_vector right_mat]; % 左右两边部分组合成Q矩阵

C = zeros(train_num,train_num);   % 初始化
for i = 1:train_num
   C = C + (all_S_matrix{1,i}) * Q_matrix * ((all_S_matrix{1,i})');
end

% 获得Ω矩阵                       

Res = cell(1,2);
Res{1,1} = eye(I(1));     % 初始化 U1∈ R（I1 × I1 = 300 × 300）
Res{1,2} = eye(I(2));   % 初始化 U2∈ R（I2 × I2 = 5  × 5）

P = cell(1,2);
P{1,1} = J(1);   % 降维后的维度P1，论文中取特征值小于0的个数为Pk
P{1,2} = J(2);    % 降维后的维度P2

%%********************************************************************************
%%***************************6·目标函数 F(k)**************************************
%%********************************************************************************
accept_error = 1 ;             % 收敛条件：2次迭代误差小于 1
time = 7;                      % 迭代次数
lamda_array = zeros(1,time);   % 判别收敛条件的 lamda
for k = 1:time
    for i = 1:2
        % 6.1 求矩阵F_K 【...TODO....OK】
        F_K = zeros(size(Res{1,i},1),size(Res{1,i},1));    % F_K需要用到前面状态的F_K，因此每次求之前都要初始化为0矩阵，以免被污染
        for g = 1:train_num           % 求F_K 矩阵
            if i == 1
                temp1 = ttm(train_set(:,:,g),(Res{1,2})',2);
            else  % i == 2
                temp1 = ttm(train_set(:,:,g),(Res{1,1})',1);
            end
            for h = 1:train_num
                if i == 1
                    temp2 = ttm(train_set(:,:,h),(Res{1,2})',2);
                else % i == 2
                    temp2 = ttm(train_set(:,:,h),(Res{1,1})',1);
                end
                t1 = tenmat(temp1,i);
                t2 = tenmat(temp2,i);
                temp_mat = C(g,h) * t1.data * ((t2.data)');
                F_K = F_K + temp_mat;
            end
        end
        % 6.2 求F_K的特征值和特征向量，把最小的Pk个特征向量组合成U
        [V,D] = eig(F_K); % V是特征向量
        D = diag(D); % D是一个存放特征值的一维数组
        [D_sort,D_index] = sort(D,'ascend'); % 排序，D_sort是排序后的特征值，D_index是排序的原序号
        V_sort=V(:, D_index); % V_sort就是对应排序后的特征向量 【升序】
        if i == 1 
            Res{1,i} = zeros(I(1),P{1,i});
        else
            Res{1,i} = zeros(I(2),P{1,i});
        end
        for j=1:P{1,i}
            Res{1,i}(:,j) = V_sort(:,j);
        end
    end
    
% 计算收敛lambda   
    lamda = 0;
    for g = 1: train_num
        temp3 = ttm(ttm(train_set(:,:,g),(Res{1,1})',1),(Res{1,2})',2);
       for h = 1:train_num
           temp4 = ttm(ttm(train_set(:,:,h),(Res{1,1})',1),(Res{1,2})',2);
           lamda = lamda + C(g,h) * innerprod(temp3,temp4);     
       end
    end
    lamda_array(k) = lamda;
    
% 计算和判断收敛条件
    if k ~= 1                             % 以后每来一个数，计算error并判断是否小于accept_error 小于则证明收敛了
        error = abs(lamda_array(k) - lamda_array(k-1));
        if error <= accept_error
            break;
        end
    end
end
ela=cputime - ela;
Us = Res;
end

