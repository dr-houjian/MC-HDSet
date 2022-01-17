%This is used to generate the hsima used in hdset matching between images.

function [hsima cmatch]=hdset2hsima_img(P1,P2,descr1,descr2,nT,nNN0,scale,th_sim)

    if size(P1,1)==2
        P1=P1';
    end
    if size(P2,1)==2
        P2=P2';
    end
    
    %triangles
    [t1,t2,feat_t1,feat_t2]=pts2tri(P1,P2,nT);

    %sparsify
    nP2=size(P2,1);
    nT2=nP2*(nP2-1)*(nP2-2);
    if nNN0<=1
        nNN=floor(nT2*nNN0);
    else
        nNN=min(nT2,nNN0);
    end
    nNN=max(nNN,5);
    [ct1 ct2 hsim]=tri2ct_img(feat_t1,feat_t2,descr1,descr2,t1,t2,nNN,scale,th_sim);
    
    %hsima
    nP1=size(P1,1);
    nP2=size(P2,1);
    [hsima,cmatch]=ct2hsima(ct1,ct2,hsim,nP1,nP2);
  
end

function [t1,t2,feat_t1,feat_t2]=pts2tri(P1,P2,nT)

    %generate triangles
    t1=generate_t1(P1,nT);
    t2=generate_t2(P2);
    
    %generate descriptions of triangles
    feat_t1=feature_sin(t1,P1);
    feat_t2=feature_sin(t2,P2);

end

%This is used to generate the triangles in a set of data, used in
%hypergraph matching.
function t1=generate_t1(P1,nT)  %revise to be the triangles between nearest points or farthest points

    if ~exist('nT','var')
        nT=1;
    end
    
    nP1=size(P1,1);
    
    if nT<=1
        npts=round((nP1-1)*nT);
    else
        npts=min(nP1-1,nT);
    end

    dima=slmetric_pw(P1',P1','eucdist');
    
    [~,idx_sort]=sort(dima,2,'ascend');
    t1=zeros(round(nP1*npts*(npts-1)/2),3);
    
    count=0;
    for i=1:nP1
        ss=idx_sort(i,2:npts+1);
        t10=combntns(ss,2);
        nt10=size(t10,1);
        t11=ones(nt10,1)*i;
        t12=[t11,t10];
        t1(count+1:count+nt10,:)=t12;
        count=count+nt10;
    end
    
    t1=sort(t1,2,'ascend');
    t1=unique(t1,'rows');
    
end

%This is used to generate the triangles in reference image, used in
%hypergraph matching.
function t2=generate_t2(P2)

    nP2=size(P2,1);
    
    ss=1:nP2;
    t2=combntns(ss,3);

end

function feat=feature_sin(tri,P)

    nt=size(tri,1);
    feat=zeros(nt,3);
    for i=1:nt
        vec=tri(i,:);
        v1=P(vec(1),:);
        v2=P(vec(2),:);
        v3=P(vec(3),:);
        d12=sqrt(sum((v1-v2).^2));
        d23=sqrt(sum((v2-v3).^2));
        d13=sqrt(sum((v1-v3).^2));
                
        cos1=(d12^2+d13^2-d23^2)/(2*d12*d13);
        cos2=(d12^2+d23^2-d13^2)/(2*d12*d23);
        cos3=(d13^2+d23^2-d12^2)/(2*d13*d23);
                
        feat(i,:)=[(1-cos1^2)^0.5 (1-cos2^2)^0.5 (1-cos3^2)^0.5];
    end

end

%This is to build the similarity between triangles.
function [ct1 ct2 hsim]=tri2ct_img(feat_tri1,feat_tri2,feat_vertex1,feat_vertex2,t1,t2,nNN,sigma,th_sim)

    if size(feat_tri1,1)<size(feat_tri1,2)
        feat_tri1=feat_tri1';
    end
    if size(feat_tri2,1)<size(feat_tri2,2)
        feat_tri2=feat_tri2';
    end
    
    t22=t2(:,[1 3 2]);
    t23=t2(:,[2 1 3]);
    t24=t2(:,[2 3 1]);
    t25=t2(:,[3 1 2]);
    t26=t2(:,[3 2 1]);
    t2=[t2;t22;t23;t24;t25;t26];
    
    feat_tri22=feat_tri2(:,[1 3 2]);
    feat_tri23=feat_tri2(:,[2 1 3]);
    feat_tri24=feat_tri2(:,[2 3 1]);
    feat_tri25=feat_tri2(:,[3 1 2]);
    feat_tri26=feat_tri2(:,[3 2 1]);
    feat_tri2=[feat_tri2;feat_tri22;feat_tri23;feat_tri24;feat_tri25;feat_tri26];
    
    [inds, dists] = annquery(feat_tri2', feat_tri1', nNN, 'eps', 10);
    inds=inds';
    dists=dists';
    sigma=mean(dists(:))*sigma;
   
    %build hsim
    nt1=size(t1,1);
    ct1=zeros(nt1*nNN,3);
    ct2=zeros(nt1*nNN,3);
    hsim=zeros(nt1*nNN,1,'single');
    
    sima_sift=slmetric_pw(single(feat_vertex1'),single(feat_vertex2'),'nrmcorr');
    
    count=0;
    for i=1:nt1
        ct10=t1(i,:);
        for j=1:nNN
            ct20=t2(inds(i,j),:);
            
            sim1=sima_sift(ct10(1),ct20(1));
            sim2=sima_sift(ct10(2),ct20(2));
            sim3=sima_sift(ct10(3),ct20(3));
            
            if sim1>th_sim && sim2>th_sim && sim3>th_sim
                count=count+1;
                ct1(count,:)=ct10(:);
                ct2(count,:)=ct20(:);
                hsim(count)=exp(-dists(i,j)^2/sigma^2);
            end
        end
    end
    clear sima_sift;
    
    ct1=ct1(1:count,:);
    ct2=ct2(1:count,:);
    hsim=hsim(1:count,:);
end

%This is to build hsima based on the candiate triangles and similarities.
function [hsima,cmatch]=ct2hsima(ct1,ct2,hsim,nP1,nP2)

    [nct,order]=size(ct1);    %number of candidate triangle matches
    
    cmatch=zeros(nP1*nP2,2);
    k=1;
    for i=1:nP1
        for j=1:nP2
            cmatch(k,:)=[i,j];
            k=k+1;
        end
    end
    
    %obtain all the candidate matches and build the hyper-sima
    hsima=zeros(nct,order+1,'single'); %the idx of the three candidiate matches, and corresponding sim
    
    idx=zeros(1,order);       %the idx of the three candidate matches in a triangle in cmatch
    for i=1:nct               %all the candidate triangle matches
        for j=1:order         %all the candiate matches in a triangle
            idx(j)=(ct1(i,j)-1)*nP2+ct2(i,j);
        end
        hsima(i,1:order)=idx;
        hsima(i,order+1)=hsim(i);
    end

    clear hsim ct;
end
