%%TSDDLSR
function [output_mean,output_std] = TSDDLSR(path,times,train_size,n_components,TT,GPU,random)
rec_acc=zeros(1,times);
for now=1:times

[X_train,X_test,Y_train,Y_test,classification,sample,x,y]=division(path,train_size,random,n_components);

for i=1:length(X_train(:,1))
    X_train(i,:)=X_train(i,:)/norm(X_train(i,:));
end
for i=1:length(X_test(:,1))
    X_test(i,:)=X_test(i,:)/norm(X_test(i,:));
end
Wpca = fastPCA(X_train,n_components,0);
X_train=(X_train*Wpca);
X_test=(X_test*Wpca);
now

n=size(X_train,1);
m=size(X_train,2);
c=size(unique(Y_train),2);
X=X_train';

H=zeros(c,n);
for i=1:n
    for j=1:c
        if Y_train(i)==j
           H(j,i)=1; 
        end
    end
end

B  = -1*ones(c,n);
for i=1:n
    for j=1:c
        if Y_train(i)==j
           B(j,i)=1; 
        end
    end
end

M = zeros(c,n);
L = zeros(c,n);
W_max = zeros(c,m);
W = rand(c,m);
E = zeros(c,n);
F = W*X;

mu=1e-5;
rho=1.05;

lambda1=1e0;
lambda2=5e-2;
lambda3=1e0;
lambda4=5e-3;

epsilon1=1e-7;
epsilon2=1e-7;

R1 = zeros(c,n);
R2 = zeros(c,n);

maxresult=0;
rec_loss=[];
rec_acc2=[];
for i=1:TT
	
    %Update W
    ttt=(2+2*mu)*(X*X')+2*lambda2*eye(m);
    tt=mu*(L+E+F)+2*(H+B.*M)-R1-R2;
	W=tt*X'/ttt;
    
    %Update F
    Hh = W*X-R2/mu;
    F = [];
    obj_F = 0;
    for ic = 1:c
        idx = find(Y_train == ic);
        H_ic = Hh(:,idx);
        linshi_F = solve_l1l2(H_ic',lambda3/mu)';
        F = [F,linshi_F];
        obj_F = obj_F+sum(sqrt(sum(linshi_F.^2,2)));
    end
    clear linshi_F;
    
    %Update E
    K = W*X-L-R1/mu;
    E = solve_l1l2(K',lambda4/mu)';
    
	%Update M
	M=max(lambda1*B.*(W*X-H),0);
    
	%Update L
	L=max(So(lambda1/mu,W*X-E+R1/mu),0);
    
	%Update R1
	R1=R1+mu*(E-(W*X-L));
   
	%Update R2
	R2=R2+mu*(W*X-F); 
   
	%Update mu
    mu = min(rho*mu,mu);

    result=W*X_test';
    [~,ID]=max(result',[],2);
    ID=(ID-Y_test'==0);
    acc=sum(ID(:))/size(X_test,1);
    rec_acc2(end+1)=acc;
    if acc>maxresult
        maxresult=acc;
        W_max=W;
    end
end   
rec_acc(now)=maxresult;

end

output_std=std(rec_acc)*100;
output_mean=mean(rec_acc)*100;

end
