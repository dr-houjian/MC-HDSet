% This is the code of the MC-HDSet matching algorithm proposed in
% Jian Hou, Marcello Pelillo, Huaqiang Yuan. Hypergraph matching via
% game-theoretic hypergraph clustering. Pattern Recognition, vol. 125, 2022.

function demo_mchdset()

    %parameters
    nT=0.5;
    nNN=500;
    th_sim=0.8;
    sigma=10;
    
    %read image and features
    fimg1='img1.ppm';
    fimg2='img2.ppm';
    img1=imread(fimg1);
    img2=imread(fimg2);
    
    para.th_p1=15;
    para.th_e1=3;
    para.th_p2=15;
    para.th_e2=3;
    para.th_d=10;
    [P1,P2,descr1,descr2]=img2feat(img1,img2,para);
    
    %build hsima
    [hsima cmatch]=hdset2hsima_img(P1,P2,descr1,descr2,nT,nNN,sigma,th_sim);
    
    %do clustering
    label=hsima2match_mc(hsima,cmatch);
    
    %matches
    ncluster=max(label);
    match_c=cell(1,ncluster);
    for i=1:ncluster
        idx_hdset=label==i;
        match=cmatch(idx_hdset,:);
        match_c(i)={match};
    end
    
    %show the matches
    disp_match_multi(match_c,img1,img2,P1,P2,1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%This is detect features and descriptors.
function [P1,P2,descr1,descr2]=img2feat(img1,img2,para)

th_p1=para.th_p1;
th_e1=para.th_e1;
th_p2=para.th_p2;
th_e2=para.th_e2;
th_d=para.th_d;

[height1,width1]=size(img1);
[height2,width2]=size(img2);

[P1,descr1]=img2hmp(img1,th_p1,th_e1);
[P2,descr2]=img2hmp(img2,th_p2,th_e2);
[P1,descr1]=pts_filter(P1,descr1,th_d,height1,width1);
[P2,descr2]=pts_filter(P2,descr2,th_d,height2,width2);

end

function [pts,descr]=img2hmp(img,th_p,th_e)

    nchannel=size(img,3);
    if nchannel==3
        img=single(rgb2gray(img));
    else
        img=single(img);
    end
    
    %SIFT feature extraction with the vlfeat library
    [frames,descr]=vl_sift(img,'PeakThresh',th_p,'EdgeThresh',th_e);
    pts=frames(1:2,:);
    pts=pts';
    descr=descr';

    %post-processing, remove the features in the same position
    npts=size(pts,1);
    for i=1:npts-1
        for j=i+1:npts
            if i<=npts && j<=npts && sum(abs(pts(i,:)-pts(j,:)))==0
                pts(j,:)=[];
                descr(j,:)=[];
                j=j-1;
                npts=size(pts,1);
            end
        end
    end
end

%sparsify the features
function [P1,descr1]=pts_filter(P1,descr1,th,height,width)

    i=1;
    while 1>0
        pt1=P1(i,:);
        
        j=i+1;
        while 1>0
            pt2=P1(j,:);
            d=sqrt(sum((pt1-pt2).^2));
            
            if d<th
                P1(j,:)=[];
                descr1(j,:)=[];
                j=j-1;
            end
            
            j=j+1;
            
            if j>size(P1,1)
                break;
            end
        end
        
        i=i+1;
        
        if i>=size(P1,1)
            break;
        end
    end
    
    i=1;
    while 1>0,
        if P1(i,1)<20 || P1(i,2)<20 || P1(i,1)>width-20 || P1(i,2)>height-20
            P1(i,:)=[];
            descr1(i,:)=[];
        end
        i=i+1;
        
        if i>size(P1,1)
            break;
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
function t1=generate_t1(P1,nT)

    if ~exist('nT','var')
        nT=1;
    end
    
    nP1=size(P1,1);
    
    if nT<=1
        npts=round((nP1-1)*nT);
    else
        npts=min(nP1-1,nT);
    end

    dima=squareform(pdist(P1,'euclidean'));
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
    
    sima_sift=getCosSim(single(feat_vertex1),single(feat_vertex2));

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

function sima=getCosSim(x,y)

    if size(x,2)~=size(y,2)
        printf('Error! Each row must correspond to a data vector!');
    else
        n1=size(x,1);
        n2=size(y,1);
        sima=zeros(n1,n2);
        
        for i=1:n1
            x1=x(i,:);
            for j=1:n2
                y1=y(j,:);
                xy=dot(x1,y1);
                nx=norm(x1);
                ny=norm(y1);
                sima(i,j)=xy/(nx*ny);
            end
        end
    end

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%This is to extract matches with hdset.
function label=hsima2match_mc(hsima,cmatch)

    th_weight=0.0001;
    toll=1e-3;
    max_iter=500;
    
    [nedge,dimen]=size(hsima);
    ndata=size(cmatch,1);
    ncardi=dimen-1;
    
    H.nVertices=ndata;
    H.nEdges=nedge;
    H.edges=hsima(:,1:dimen-1);
    H.w=hsima(:,dimen);
    
    label=zeros(1,ndata);
    
    x=ones(ndata,1);
    x=x/sum(x);

    nhdset=0;
    while 1>0
        %do clustering
        x=hgtExtractCluster(H,x,toll,max_iter);
        x=reshape(x,1,length(x));
        
        %enforce intra-cluster one-to-one constraint
        x=oto2x(x,cmatch);
        
        %the cluster
        idx_hdset=find(x>th_weight);
        nhdset=nhdset+1;
        label(idx_hdset)=nhdset;
        
        if nhdset>1
            %enforce inter-cluster one-to-one constraint
            label=oto2label(label,cmatch);
            
            idx_cluster=find(label==nhdset);
            if length(idx_cluster)<4
                label(idx_cluster)=0;
                break;
            end
        end
        
        if nhdset==3
            break;
        end
        
        %prepare for the next, new hsima and new x
        for i=idx_hdset
            row= hsima(:,1:ncardi)==i;
            hsima(row,ncardi+1)=0;
        end
        
        idx_out= label==0;
        x=zeros(ndata,1);
        x(idx_out)=1;
        x=x/sum(x);
    end

end

function x=oto2x(x,cmatch)

    if size(x,1)>1
        x=x';
    end

    [~,idx_sort]=sort(x,'descend');
    vec_t=cmatch(idx_sort,1);
    
    vt=unique(cmatch(:,1));
    for i=1:length(vt)
        no=vt(i);
        idx=find(vec_t==no);
        if length(idx)>1
            x(idx_sort(idx(2:length(idx))))=0;
        end
    end
    
    vec_r=cmatch(idx_sort,2);
    vr=unique(cmatch(:,2));
    for i=1:length(vr)
        no=vr(i);
        idx=find(vec_r==no);
        if length(idx)>1
            x(idx_sort(idx(2:length(idx))))=0;
        end
    end

end

function label=oto2label(label,cmatch)

    max_label=max(label);
    
    for k=1:max_label-1
        idx_c1=find(label==k);
        idx_c2=find(label>k);
    
        for i=idx_c1
            for j=idx_c2
                if cmatch(i,1)==cmatch(j,1)
                    label(j)=0;              %matches in larger-sequence-number clusters are discarded
                end
                if cmatch(i,2)==cmatch(j,2)
                    label(j)=0;
                end
            end
        end
    end

end

% This is the code of
%
% S. R. Bulo, M. Pelillo, A game-theoretic approach to hypergraph clustering,
% IEEE Transactions on Pattern Analysis and 755 Machine Intelligence 35 (6)
% (2013) 1312¨C1327.
%
% Written by Samuel Rota Bulo.
%
% Hypergraph clustering
% 
% H: hypergraph structure
%    H.nVertices: number of vertices (N)
%    H.nEdges:    number of edges (M)
%    H.edges:     MxK matrix of indices to vertices representing the edges
%    H.w:         Mx1 vector edge weights
%
% x: Nx1 starting point in the simplex
%
% maxiter: maximum number of iterations

function [x,Fx,niter]=hgtExtractCluster(H,x,toll,maxiter)
  if ~exist('maxiter','var')
    maxiter=1000;
  end
  
  if size(x,2)>1
      x=x';
  end
  
  niter=0;
  error=2*toll;
  old_x=x;
  while niter<maxiter && error>toll
    Fx=zeros(H.nVertices,1);
    for i=1:H.nEdges
      edge=H.edges(i,:);
      tmp=prod(x(edge))*H.w(i);
      if tmp>0
        Fx(edge)=Fx(edge)+tmp./x(edge);
      end 
    end
    x=x.*Fx;
    xFx=sum(x);
    if xFx==0
      return;
    end
    x=x/xFx;
    
    error=norm(x-old_x);
    old_x=x;
    
    niter=niter+1;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is to display different parts of matches in different colors.
function disp_match_multi(match_c,img1,img2,P1,P2,flag_combine)

    if ~exist('flag_combine','var')
        flag_combine=1;
    end
    
    nlabel=length(match_c);
    match=[];
    for i=1:nlabel
        match=[match;match_c{i}];
    end
    label=zeros(1,size(match,1));

    npart=length(match_c);
    count=0;
    for i=1:npart
        match0=match_c{i};
        nmatch0=size(match0,1);
        label(count+1:count+nmatch0)=i;
        count=count+nmatch0;
    end
    
    nLineWidth=3;
    
    color=cell(1,7);
    color(1)={'b'};
    color(2)={'r'};
    color(3)={'g'};
    color(4)={'k'};
    color(5)={'m'};
    color(6)={'y'};
    color(7)={'c'};
    
    line=cell(1,3);
    line(1)={'-'};
    line(2)={'--'};
    line(3)={'-.'};
    
    %the combined image
    img3=img_combine(img1,img2,flag_combine);
    [h1,w1,~]=size(img1);
    
    imshow(img3,'Border','tight');
    hold on;
    
    %the feature points
    alpha=0:pi/20:2*pi;
    R=6;
    x0=R*cos(alpha);
    y0=R*sin(alpha);
    
    for i=1:size(P1,1)
        x=P1(i,1);
        y=P1(i,2);
        
        x1=x+x0;
        y1=y+y0;
        plot(x1,y1,'r-','LineWidth',1);
    end
    
    for i=1:size(P2,1)
        if flag_combine==1
            x=P2(i,1)+w1;
            y=P2(i,2);
        else
            x=P2(i,1);
            y=P2(i,2)+h1;
        end
        
        x1=x+x0;
        y1=y+y0;
        plot(x1,y1,'r-','LineWidth',1);
    end

    %matches
    alpha=0:pi/20:2*pi;
    R=6;
    x0=R*cos(alpha);
    y0=R*sin(alpha);
    
    for i=1:size(match,1)
        k=label(i);
        ii=floor((k-1)/7);
        jj=round(k-ii*7);
        ii=round(ii+1);
        symbol=[color{jj},line{ii}];
        
        idx1=match(i,1);
        idx2=match(i,2);
        
        if idx1==0 || idx2==0
            continue;
        end
        
        %feature
        x1=P1(idx1,1);
        y1=P1(idx1,2);
        plot(x1+x0,y1+y0,'r-','LineWidth',1);
        fill(x1+x0,y1+y0,'r');
        
        %match
        if flag_combine==1
            x2=P2(idx2,1)+w1;
            y2=P2(idx2,2);
        else
            x2=P2(idx2,1);
            y2=P2(idx2,2)+h1;
        end
        
        plot(x2+x0,y2+y0,'r-','LineWidth',1);
        fill(x2+x0,y2+y0,'r');
    
        plot([x1,x2],[y1,y2],symbol,'LineWidth',nLineWidth);
        hold on;
    end
    
    axis off;
    hold off;
    
    aframe=getframe(gcf);
    imwrite(aframe.cdata,'d:\match.jpg');

end

%This is used to merge two images into one, used to show matching results.
function img=img_combine(img1,img2,mode)

    if ~exist('mode','var')
        mode=1;
    end

    [h1,w1,nc1]=size(img1);
    [h2,w2,nc2]=size(img2);

    if mode==1
        if h1>h2
            img20=zeros(h1,w2,nc2,'uint8');
            img20(1:h2,1:w2,1:nc2)=img2;
            img=[img1,img20];
        elseif h1<h2
            img10=zeros(h2,w1,nc1,'uint8');
            img10(1:h1,1:w1,1:nc1)=img1;
            img=[img10,img2];
        else
            img=[img1,img2];
        end
    else
        if w1>w2
            img20=zeros(h2,w1,nc2,'uint8');
            img20(1:h2,1:w2,1:nc2)=img2;
            img=[img1;img20];
        elseif w1<w2
            img10=zeros(h1,w2,nc1,'uint8');
            img10(1:h1,1:w1,1:nc1)=img1;
            img=[img10;img2];
        else
            img=[img1;img2];
        end
    end

end