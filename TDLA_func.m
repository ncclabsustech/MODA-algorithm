function [Us, ela] = TDLA_func(train,train_label,tensor_shape,lowerdims,k1,k2)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
ela = cputime;
[~, N] = size(train);
train_num = N;
I = tensor_shape;
J = lowerdims; %%%%%%%%%%%%%%%%%%%%%NOT SURE
X = reshape( train,[I(1),I(2),N]);
train_set = tensor(X,[I(1),I(2),N]);
%******************************ѡ�����Si********************                           % k1��ͬ��ĺ�k2����ͬ��ģ��Լ���������Xi������ϳ� Patch(Xi)
all_S_matrix = cell(1,train_num);             % �������ѵ��������ѡ���������ݽṹ   
for i = 1:train_num                           % Si�����ÿ��Xi��������  N��(n+1) ÿ������i��N�� n+1 ��  Si(a,b) ��ʾ��b�е�a�У�
                                                % ���ֵΪ1������N�������е�a������Ϊ����Xi��Patch�еĵ�a������
    sample_E_dis = zeros(1,train_num);        % ����train_set{1,i} ������������ŷʽ�������� sample_E_dis�У������������n1��n2 ����Si���󣬷ŵ�all_S_matrix��
    select_matrix = zeros(train_num,k1+k2+1);
    for j = 1:train_num
        re = train_set(:,:,i) - train_set(:,:,j);
        re = tenmat(re,1);
        re = re.data;
        re = re.^2;
        re = sum(re(:));
        sample_E_dis(1,j) = sqrt(re);           % sample_E_dis ��ʾ��ǰ�������������������������Լ�����ŷʽ����
    end
    % end for ������Xi����������������ŷʽ����sample_E_dis(1,sam_num)
    % [sample_E_dis,~] = mapminmax(sample_E_dis,0,1);
    [~,pos_E_dis] = sort(sample_E_dis,'ascend');
    
    select_matrix(pos_E_dis(1,1),1)=1;          % ѡ������һ�п϶���ŷʽ����Ϊ0����������
    
    ind = 2;
    for k=2:(k1+1)                              % [2,...,n1+1]��n1��������ͬ��Ľ������
        while train_label(1,pos_E_dis(1,ind)) ~= train_label(1,pos_E_dis(1,1))  % ��ͬ����ֱ������
            ind = ind + 1;
        end
        select_matrix(pos_E_dis(1,ind),k) = 1;   % select_matrix(i,j) = 1 ��ʾ��Xi��patch�� ��j��λ�����ض�Ӧ��������sample_cell�ĵ�i��λ��  
        ind = ind + 1;
    end
    % end for ��n1 ��ͬ��������������ı�ţ�����Xi��ѡ�����Si�ж�Ӧ�е�λ�ñ�Ϊ1
    
    ind = 2;
    for k = (k1+2):(k1+k2+1)                     % [n1+2,...,n1+n2+1]��n2����������ͬ��Ľ������
        while train_label(1,pos_E_dis(1,ind)) == train_label(1,pos_E_dis(1,1))  % ��ŷʽ������С����ͬ���������
           ind = ind + 1;
        end
        select_matrix(pos_E_dis(1,ind),k) = 1;
        ind = ind + 1;
    end
    %end for ��n2 ����ͬ��������������ı�ţ�����Xi��ѡ�����Si�ж�Ӧ�е�λ�ñ�Ϊ1
    
    all_S_matrix{1,i} = select_matrix;   %��Xi��ѡ�����ŵ��ܵ�ѡ�������ȥ
end


%*********���������е�Q����*********
s_weight = 1;                       % patch ��ͬ��������Ȩ�� = ������Ϊ1     
dif_weight = -4;                    % patch �в�ͬ�������е�Ȩ�� = �����е�a   ��= [1,1,1,1,1,a,a,a,a,a]
Belta = [s_weight*ones(k1,1);dif_weight*ones(k2,1)];
Belta = Belta;
sum_Belta = sum(Belta);
left_vector = [sum_Belta;-Belta];   % Q����߲���
right_mat = [-Belta';diag(Belta)];  % Q���ұ߲���
Q_matrix = [left_vector right_mat]; % �������߲�����ϳ�Q����

C = zeros(train_num,train_num);   % ��ʼ��
for i = 1:train_num
   C = C + (all_S_matrix{1,i}) * Q_matrix * ((all_S_matrix{1,i})');
end

% ��æ�����                       

Res = cell(1,2);
Res{1,1} = eye(I(1));     % ��ʼ�� U1�� R��I1 �� I1 = 300 �� 300��
Res{1,2} = eye(I(2));   % ��ʼ�� U2�� R��I2 �� I2 = 5  �� 5��

P = cell(1,2);
P{1,1} = J(1);   % ��ά���ά��P1��������ȡ����ֵС��0�ĸ���ΪPk
P{1,2} = J(2);    % ��ά���ά��P2

%%********************************************************************************
%%***************************6��Ŀ�꺯�� F(k)**************************************
%%********************************************************************************
accept_error = 1 ;             % ����������2�ε������С�� 1
time = 7;                      % ��������
lamda_array = zeros(1,time);   % �б����������� lamda
for k = 1:time
    for i = 1:2
        % 6.1 �����F_K ��...TODO....OK��
        F_K = zeros(size(Res{1,i},1),size(Res{1,i},1));    % F_K��Ҫ�õ�ǰ��״̬��F_K�����ÿ����֮ǰ��Ҫ��ʼ��Ϊ0�������ⱻ��Ⱦ
        for g = 1:train_num           % ��F_K ����
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
        % 6.2 ��F_K������ֵ����������������С��Pk������������ϳ�U
        [V,D] = eig(F_K); % V����������
        D = diag(D); % D��һ���������ֵ��һά����
        [D_sort,D_index] = sort(D,'ascend'); % ����D_sort������������ֵ��D_index�������ԭ���
        V_sort=V(:, D_index); % V_sort���Ƕ�Ӧ�������������� ������
        if i == 1 
            Res{1,i} = zeros(I(1),P{1,i});
        else
            Res{1,i} = zeros(I(2),P{1,i});
        end
        for j=1:P{1,i}
            Res{1,i}(:,j) = V_sort(:,j);
        end
    end
    
% ��������lambda   
    lamda = 0;
    for g = 1: train_num
        temp3 = ttm(ttm(train_set(:,:,g),(Res{1,1})',1),(Res{1,2})',2);
       for h = 1:train_num
           temp4 = ttm(ttm(train_set(:,:,h),(Res{1,1})',1),(Res{1,2})',2);
           lamda = lamda + C(g,h) * innerprod(temp3,temp4);     
       end
    end
    lamda_array(k) = lamda;
    
% ������ж���������
    if k ~= 1                             % �Ժ�ÿ��һ����������error���ж��Ƿ�С��accept_error С����֤��������
        error = abs(lamda_array(k) - lamda_array(k-1));
        if error <= accept_error
            break;
        end
    end
end
ela=cputime - ela;
Us = Res;
end

