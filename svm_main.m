clear;
%% set parameter
C = 100000; % soft or hard margin
P = 2; % polynomial value
m1 = 1.2; % kernel parameter
m2 = 1.5;
%% load data
train = load('train.mat');
test = load('eval.mat');

train_data = train.train_data;
train_label = train.train_label;
[feature N] = size(train.train_data);

test_data = test.eval_data;
test_label = test.eval_label;
[A M] = size(test_data);

%% optimization
K = new_kernel(train_data,train_data,m1,m2,P);
H = (train_label*train_label').*K;
f = -ones(N,1);
Aeq = train_label';
beq = 0;
lb = zeros(N,1);
ub = C*ones(N,1);
options = optimset('LargeScale', 'off', 'MaxIter', 1000);
x = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],options);

%% find support vector
if(C<1e6)
    index = find(x>1e-4); % Soft margin
else
    index = find(x==max(x)); % hard margin
end

%% hyperplane
W_x = zeros(N,1);
[O Q] = size(index);
for i=index
    for j=1:N
        W_x(i) = W_x(i) +  x(j,1) * train_label(j,1) * K(i,j);
    end
    B(i) = train_label(i,1) - W_x(i);
end
B(B==0) = [];
b = mean(B);

%% test
W_x_t = zeros(N,1);
K_t = new_kernel(test_data,train_data,m1,m2,P);
for i=1:M
    for j=1:N
        W_x_t(i) = W_x_t(i) +  x(j,1) * train_label(j,1) * K_t(i,j);      
    end
    eval_predicted(i) = sign(W_x_t(i)+b);
end
[O Q] = size(find((test_label'-eval_predicted)==0));
accuracy_eval = Q/M

W_x = zeros(N,1);
for i=1:N
    for j=1:N
        W_x(i) = W_x(i) +  x(j,1) * train_label(j,1) * K(i,j);      
    end
    label_x(i) = sign(W_x(i)+b);
end
[O P] = size(find((train_label'-label_x)==0));
accuracy_train = P/N
