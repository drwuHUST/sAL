%% Implementation of the AL algorithms in:
%%
%% D. Wu, "Pool-based sequential active learning for regression," IEEE Trans. on Neural 
%% Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% Compare 10 methods:
%% 1. BL
%% 2. EMCM
%% 3. RD + EMCM
%% 4. QBC
%% 5. RD + QBC
%% 6. GS
%% 7. RD + GS
%% 8. RD only
%% 9. EQBC
%% 10. EEMCM

%% Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all;   rng('default');
datasets={'Concrete-CS','Yacht'};
nRepeat=10; % number of repeats to get statistically significant results;  100 was used in the paper
rr=.01; % RR parameter
nBoots=4;
numAlgs=10;
MSEs=cell(1,length(datasets)); CCs=MSEs;

for s=1:length(datasets)   
    temp=load([datasets{s} '.mat']); data=temp.data;
    X0=data(:,1:end-1); Y0=data(:,end); X0=zscore(X0);
    numY0=length(Y0);
    minN=min(20,size(X0,2)+1); % mininum number of training samples
    maxN=min(60,max(20,ceil(.1*numY0))); % maximum number of training samples
    MSEs{s}=nan(numAlgs,maxN,nRepeat); CCs{s}=MSEs{s};
    
    %% Pre-compute distances for GS
    distX0=zeros(numY0);
    for i=1:numY0
        M=X0-repmat(X0(i,:),numY0,1);
        distX0(:,i)=sqrt(sum(M.^2,2));
    end
    
    %% Iterative approaches; random sampling
    for r=1:nRepeat
        [s r]
        
        %% random effect; 80% samples
        ids=datasample(1:numY0,round(numY0*.8),'Replace',false);
        X=X0(ids,:); Y=Y0(ids); numY=length(Y);
        idsTest=1:numY0; idsTest(ids)=[];
        XTest=X0(idsTest,:); YTest=Y0(idsTest); numYTest=length(YTest);
        idsTrain=repmat(datasample(1:numY,maxN,'Replace',false),numAlgs,1);
        distX=distX0(ids,ids);
        
        %% For EQBC and EEMCM
        idxOutlierGroup=1; idsGood9=1:numY; numEle=zeros(1,minN);
        while ~isempty(idxOutlierGroup)
            ids=kmeans(X(idsGood9,:),minN);
            for i=1:minN
                numEle(i)=sum(ids==i);
            end
            idxOutlierGroup=find(numEle<.02*length(idsGood9));
            idsOutliers=[];
            for i=1:length(idxOutlierGroup)
                idsOutliers=cat(2,idsOutliers,find(ids==idxOutlierGroup(i)));
            end
            idsGood9(idsOutliers)=[];
        end
        idsGood10=idsGood9;
        
        for n=minN:maxN
            %% 1. BL: No AL
            b1=ridge(Y(idsTrain(1,1:n)),X(idsTrain(1,1:n),:),rr,0);
            Y1=[ones(numYTest,1) XTest]*b1;
            MSEs{s}(1,n,r)=sqrt(mean((Y1-YTest).^2));
            CCs{s}(1,n,r)=corr(Y1,YTest);
            
            %% 2. EMCM; the first minN samples were obtained randomly
            b2=ridge(Y(idsTrain(2,1:n)),X(idsTrain(2,1:n),:),rr,0);
            Y2=[ones(numYTest,1) XTest]*b2;
            MSEs{s}(2,n,r)=sqrt(mean((Y2-YTest).^2));
            CCs{s}(2,n,r)=corr(Y2,YTest);
            %% Select new samples by EMCM
            idsUnlabeled=1:numY; idsUnlabeled(idsTrain(2,1:n))=[];
            C=max(1,ceil(n*rand(nBoots,n)));
            Ys=repmat(Y,1,nBoots);
            for i=1:nBoots
                b=ridge(Y(idsTrain(2,C(i,:))),X(idsTrain(2,C(i,:)),:),rr,0);
                Ys(idsUnlabeled,i)=[ones(length(idsUnlabeled),1) X(idsUnlabeled,:)]*b;
            end
            EMCM=zeros(1,length(idsUnlabeled));
            Y2=Y; Y2(idsUnlabeled)=[ones(length(idsUnlabeled),1) X(idsUnlabeled,:)]*b2;
            for i=1:length(idsUnlabeled)
                for j=1:nBoots
                    EMCM(i)=EMCM(i)+norm((Ys(idsUnlabeled(i),j)-Y2(idsUnlabeled(i)))*X(idsUnlabeled(i),:));
                end
            end
            [~,idx]=max(EMCM);
            idsTrain(2,n+1)=idsUnlabeled(idx);
            
            %% 3. RD + EMCM
            if n==minN % select the center of the minN clusters as initialization
                [ids,~,~,D]=kmeans(X,n,'Replicates',10);
                idsP=cell(1,n);
                numP=zeros(1,n);
                for i=1:n
                    idsP{i}=find(ids==i);
                    numP(i)=length(idsP{i});
                end
                [~,ids2]=sort(numP,'descend');
                idsP=idsP(ids2);
                for k=1:n
                    dist=D(ids==ids2(k),ids2(k));
                    [~,idx]=min(dist);
                    idsTrain(3,k)=idsP{k}(idx);
                end
            end
            b3=ridge(Y(idsTrain(3,1:n)),X(idsTrain(3,1:n),:),rr,0);
            Y3=[ones(numYTest,1) XTest]*b3;
            MSEs{s}(3,n,r)=sqrt(mean((Y3-YTest).^2));
            CCs{s}(3,n,r)=corr(Y3,YTest);
            %% Select the new sample by RD + EMCM
            idsUnlabeled=1:numY; idsUnlabeled(idsTrain(3,1:n))=[];
            Y3=Y; Y3(idsUnlabeled)=[ones(length(idsUnlabeled),1) X(idsUnlabeled,:)]*b3;
            [ids,~,~,D]=kmeans(X,n+1,'Replicates',10); %%% Study the effect of n+1 and n+2
            idsP=cell(1,n+1);
            numP=zeros(1,n+1);
            for i=1:n+1
                idsP{i}=find(ids==i);
                numP(i)=length(idsP{i});
            end
            [numP,ids2]=sort(numP,'descend');
            idsP=idsP(ids2); D=D(:,ids2);
            for k=1:n+1
                if sum(ismember(idsTrain(3,1:n),idsP{k}))==0
                    Ys=repmat(Y(idsP{k}),1,nBoots);
                    for i=1:nBoots
                        b=ridge(Y(idsTrain(3,C(i,:))),X(idsTrain(3,C(i,:)),:),rr,0);
                        Ys(:,i)=[ones(numP(k),1) X(idsP{k},:)]*b;
                    end
                    EMCM=zeros(1,numP(k));
                    for i=1:numP(k)
                        for j=1:nBoots
                            EMCM(i)=EMCM(i)+norm((Ys(i,j)-Y3(idsP{k}(i)))*X(idsP{k}(i),:));
                        end
                    end
                    [~,idx]=max(EMCM);
                    idsTrain(3,n+1)=idsP{k}(idx);
                    break;
                end
            end
            
            %% 4. QBC; the first minN samples were obtained randomly
            b4=ridge(Y(idsTrain(4,1:n)),X(idsTrain(4,1:n),:),rr,0);
            Y4=[ones(numYTest,1) XTest]*b4;
            MSEs{s}(4,n,r)=sqrt(mean((Y4-YTest).^2));
            CCs{s}(4,n,r)=corr(Y4,YTest);
            %% Select new samples by QBC
            Ys=repmat(Y,1,nBoots);
            idsUnlabeled=1:numY; idsUnlabeled(idsTrain(4,1:n))=[];
            for i=1:nBoots
                b=ridge(Y(idsTrain(4,C(i,:))),X(idsTrain(4,C(i,:)),:),rr,0);
                Ys(idsUnlabeled,i)=[ones(length(idsUnlabeled),1) X(idsUnlabeled,:)]*b;
            end
            QBC=zeros(1,length(idsUnlabeled));
            for i=1:length(idsUnlabeled)
                QBC(i)=var(Ys(idsUnlabeled(i),:));
            end
            [~,idx]=max(QBC);
            idsTrain(4,n+1)=idsUnlabeled(idx);
            
            
            %% 5. RD + QBC
            if n==minN % select the center of the minN clusters as initialization
                idsTrain(5,1:n)=idsTrain(3,1:n);
            end
            b5=ridge(Y(idsTrain(5,1:n)),X(idsTrain(5,1:n),:),rr,0);
            Y5=[ones(numYTest,1) XTest]*b5;
            MSEs{s}(5,n,r)=sqrt(mean((Y5-YTest).^2));
            CCs{s}(5,n,r)=corr(Y5,YTest);
            %% Select the new sample by RD + QBC
            for k=1:n+1
                if sum(ismember(idsTrain(5,1:n),idsP{k}))==0
                    Ys=repmat(Y(idsP{k}),1,nBoots);
                    for i=1:nBoots
                        b=ridge(Y(idsTrain(5,C(i,:))),X(idsTrain(5,C(i,:)),:),rr,0);
                        Ys(:,i)=[ones(numP(k),1) X(idsP{k},:)]*b;
                    end
                    QBC=zeros(1,numP(k));
                    for i=1:numP(k)
                        QBC(i)=var(Ys(i,:));
                    end
                    [~,idx]=max(QBC);
                    idsTrain(5,n+1)=idsP{k}(idx);
                    break;
                end
            end
            
            
            %% 6. GS
            if n==minN
                dist=mean(distX,2);
                [~,idsTrain(6,1)]=min(dist);
                idsUnlabeled=1:numY; idsUnlabeled(idsTrain(6,1))=[];
                for i=2:n
                    dist=min(distX(idsUnlabeled,idsTrain(6,1:i-1)),[],2);
                    [~,idx]=max(dist);
                    idsTrain(6,i)=idsUnlabeled(idx);
                    idsUnlabeled(idx)=[];
                end
            end
            b6=ridge(Y(idsTrain(6,1:n)),X(idsTrain(6,1:n),:),rr,0);
            Y6=[ones(numYTest,1) XTest]*b6;
            MSEs{s}(6,n,r)=sqrt(mean((Y6-YTest).^2));
            CCs{s}(6,n,r)=corr(Y6,YTest);
            %% Select new samples by GS
            idsUnlabeled=1:numY; idsUnlabeled(idsTrain(6,1:n))=[];
            dist=min(distX(idsUnlabeled,idsTrain(6,1:n)),[],2);
            [~,idx]=max(dist);
            idsTrain(6,n+1)=idsUnlabeled(idx);
            
            %% 7. GS with new initialization
            if n==minN % select the center of the minN clusters as initialization
                idsTrain(7,1:n)=idsTrain(3,1:n);
            end
            b7=ridge(Y(idsTrain(7,1:n)),X(idsTrain(7,1:n),:),rr,0);
            Y7=[ones(numYTest,1) XTest]*b7;
            MSEs{s}(7,n,r)=sqrt(mean((Y7-YTest).^2));
            CCs{s}(7,n,r)=corr(Y7,YTest);
            %% Select the new sample by RD + GS
            for k=1:n+1
                if sum(ismember(idsTrain(7,1:n),idsP{k}))==0
                    dists=zeros(numP(k),n);
                    for i=1:n
                        M=X(idsP{k},:)-repmat(X(idsTrain(7,i),:),numP(k),1);
                        dists(:,i)=sqrt(sum(M.^2,2));
                    end
                    dist=min(dists,[],2);
                    [~,idx]=max(dist);
                    idsTrain(7,n+1)=idsP{k}(idx);
                    break;
                end
            end
            
            %% 8. RD only
            if n==minN % select the center of the minN clusters as initialization
                idsTrain(8,1:n)=idsTrain(3,1:n);
            end
            b8=ridge(Y(idsTrain(8,1:n)),X(idsTrain(8,1:n),:),rr,0);
            Y8=[ones(numYTest,1) XTest]*b8;
            MSEs{s}(8,n,r)=sqrt(mean((Y8-YTest).^2));
            CCs{s}(8,n,r)=corr(Y8,YTest);
            %% Select the new sample by RD
            for k=1:n+1
                if sum(ismember(idsTrain(8,1:n),idsP{k}))==0
                    dist=D(idsP{k},k);
                    [~,idx]=min(dist);
                    idsTrain(8,n+1)=idsP{k}(idx);
                    break;
                end
            end
            
            %% 9. EQBC
            if n==minN
                [ids,~,~,D]=kmeans(X(idsGood9,:),minN);
                idx=nan(1,minN);
                for i=1:minN
                    idx0=find(ids==i);
                    [~,j]=min(D(ids==i,i));
                    idx(i)=idx0(j);
                end
                idsTrain(9,1:n)=idsGood9(idx);
                idsGood9(idx)=[];
            end
            b9=ridge(Y(idsTrain(9,1:n)),X(idsTrain(9,1:n),:),rr,0);
            Y9=[ones(numYTest,1) XTest]*b9;
            MSEs{s}(9,n,r)=sqrt(mean((Y9-YTest).^2));
            CCs{s}(9,n,r)=corr(Y9,YTest);
            %% Select a new sample by EQBC
            Ys=repmat(Y,1,nBoots);
            for i=1:nBoots
                b=ridge(Y(idsTrain(9,C(i,:))),X(idsTrain(9,C(i,:)),:),rr,0);
                Ys(idsGood9,i)=[ones(length(idsGood9),1) X(idsGood9,:)]*b;
            end
            QBC=zeros(1,length(idsGood9));
            for i=1:length(idsGood9)
                QBC(i)=var(Ys(idsGood9(i),:));
            end
            [~,idx]=max(QBC);
            idsTrain(9,n+1)=idsGood9(idx);
            idsGood9(idx)=[];
            
            %% 10. EEMCM
            if n==minN
                idsTrain(10,1:n)=idsTrain(9,1:n); idsGood10=idsGood9;
            end
            b10=ridge(Y(idsTrain(10,1:n)),X(idsTrain(10,1:n),:),rr,0);
            Y10=[ones(numYTest,1) XTest]*b10;
            MSEs{s}(10,n,r)=sqrt(mean((Y10-YTest).^2));
            CCs{s}(10,n,r)=corr(Y10,YTest);
            %% Select a new sample by EEMCM
            idsUnlabeled=1:numY; idsUnlabeled(idsTrain(10,1:n))=[];
            Y10=Y; Y10(idsUnlabeled)=[ones(length(idsUnlabeled),1) X(idsUnlabeled,:)]*b10;
            C=max(1,ceil(n*rand(nBoots,n)));
            Ys=repmat(Y,1,nBoots);
            for i=1:nBoots
                b=ridge(Y(idsTrain(10,C(i,:))),X(idsTrain(10,C(i,:)),:),rr,0);
                Ys(idsGood10,i)=[ones(length(idsGood10),1) X(idsGood10,:)]*b;
            end
            EMCM=zeros(1,length(idsGood10));
            for i=1:length(idsGood10)
                for j=1:nBoots
                    EMCM(i)=EMCM(i)+norm((Ys(idsGood10(i),j)-Y10(idsGood10(i)))*X(idsGood10(i),:));
                end
            end
            [~,idx]=max(EMCM);
            idsTrain(10,n+1)=idsGood10(idx);
            idsGood10(idx)=[];
            
        end
    end
    
    %% Plot results
    AUCMSE=zeros(numAlgs,length(datasets)); AUCCC=AUCMSE;
    linestyle={'k-','r-','r--','g-','g--','b-','b--','k--','m-','m--'};
    ids=[1 8 4 5 2 3 6 7 9 10];
    X0=data(:,1:end-1); X0=zscore(X0); Y0=data(:,end);
    numY0=length(Y0);
    minN=size(X0,2); % mininum number of training samples
    maxN=min(60,max(20,ceil(.1*numY0))); % maximum number of training samples
    mMSEs=squeeze(nanmean(MSEs{s},3)); AUCMSE(:,s)=nansum(mMSEs,2);
    mCCs=squeeze(nanmean(CCs{s},3)); AUCCC(:,s)=nansum(mCCs,2);
    
    figure;
    set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman');
    subplot(121); hold on;
    for i=ids
        plot(minN:maxN,mMSEs(i,minN:maxN),linestyle{i},'linewidth',2);
    end
    h=legend('BL','RD','QBC','RD-QBC','EMCM','RD-EMCM','GS','RD-GS','EQBC','EEMCM','location','northeast');
    set(h,'fontsize',11);
    axis tight; box on; title(datasets{s},'fontsize',14);
    xlabel('m','fontsize',12);     ylabel('RMSE','fontsize',12);
    
    subplot(122); hold on;
    for i=ids
        plot(minN:maxN,mCCs(i,minN:maxN),linestyle{i},'linewidth',2);
    end
    h=legend('BL','RD','QBC','RD-QBC','EMCM','RD-EMCM','GS','RD-GS','EQBC','EEMCM','location','southeast');
    set(h,'fontsize',11);
    axis tight; box on; title(datasets{s},'fontsize',14);
    xlabel('m','fontsize',12);     ylabel('CC','fontsize',12);
end


