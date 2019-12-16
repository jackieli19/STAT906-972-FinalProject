%Heston LMS, dSt/St=rdt+sqrt(vt)dWt, dvt=k(theta-vt)dt+sig*sqrt(vt)dZt
A=[];
T=0.5
dt=0.5/100;
N=T/dt;
S0=100;
K=100
M=5000
v0=0.04;
r=0.05;
rho=-0.7; %correlation Wt Zt
sigma=0.1;
kappa=3;
theta=0.04;

for sed=0:99
        T=0.5
        dt=0.5/100;
        N=T/dt;
        S0=100;
        K=100
        M=5000
        v0=0.04;
        r=0.05;
        rho=-0.7; %correlation Wt Zt
        sigma=0.1;
        kappa=3;
        theta=0.04;
        randn('seed',sed)


        W=normrnd(0,1,M,N);
        WW=normrnd(0,1,M,N);
        S=zeros(M,N);
        S(:,1)=S0*ones(M,1);
        v=zeros(M,N);
        v(:,1)=v0*ones(M,1);
        %Heston Generate paths
        for jj=1:N
            S(:,jj+1)=S(:,jj).*exp((r-1/2*max(v(:,jj),0))*dt+sqrt(max(v(:,jj),0)*dt).*W(:,jj));
            v(:,jj+1)=v(:,jj)+kappa*(theta-max(v(:,jj),0))*dt+sigma*sqrt(max(v(:,jj),0)*dt).*(rho*W(:,jj)+sqrt(1-rho^2).*WW(:,jj));
        end
        Payoff=max(K-S,0);
        %3rd LMS power 3
        OptionValue=nan(M,N+1);
        OptionValue(:,N+1)=Payoff(:,N+1);
        EPut=Payoff(:,N+1);
        for ii=N:-1:2

            bina=1:M;
            X=[S(bina,ii)/K v(bina,ii)/v0];
            gprMdlb = fitrgp(X,exp(-r*dt)*OptionValue(bina,ii+1),'KernelFunction','squaredexponential');
            b=resubPredict(gprMdlb);
            gprMdlc = fitrgp(X,exp(-dt*r)*EPut(bina),'KernelFunction','squaredexponential');
            c=resubPredict(gprMdlc);
            gprMdld = fitrgp(X,exp(-dt*r)*EPut(bina).^2,'KernelFunction','squaredexponential');
            d=resubPredict(gprMdld);
            gprMdle = fitrgp(X,exp(-dt*r)*EPut(bina).*OptionValue(bina,ii+1),'KernelFunction','squaredexponential');
            e=resubPredict(gprMdle);
            theta1=-(e-(c).*(b))./(d-(c).^2);

            if sum(bina)>0
                Put = HestonPut(S(bina,ii)',K,r,sigma,(N+1-ii)*dt,v(bina,ii),kappa,theta,0,rho);
            else
                Put=[];
            end


            C=b+theta1.*(c-Put);
            C=max(C,0);
            bina2=Payoff(bina,ii)>C;
            Idx = setdiff((1:length(S)),bina(bina2));
            OptionValue(bina(bina2),ii)=Payoff(bina(bina2),ii);
            OptionValue(Idx,ii)=exp(-r*dt)*OptionValue(Idx,ii+1);
            EPut(bina(bina2))=Put(bina2);
            EPut(Idx)=exp(-r*dt)*EPut(Idx);
        end
        OV=exp(-r*dt)*OptionValue(:,2);
        EP=exp(-r*dt)*EPut;
        B=cov(exp(-r*dt)*OptionValue(:,2),EP);
        p1=exp(-r*T)*mean(Payoff(:,N+1));
        OptionV=OV-B(1,2)/sqrt(B(2,2)*B(1,1))*(EP-HestonPut(S0,K,r,sigma,T,v0,kappa,theta,0,rho));
        [PUT_1, err, ConfIntPut1]=normfit(OptionV);
        A=[A; PUT_1 err M sed S0 v0 T];
end
