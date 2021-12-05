% This is the code of the MC-HDSet matching algorithm proposed in
% Jian Hou, Marcello Pelillo, Huaqiang Yuan. Hypergraph matching via
% game-theoretic hypergraph clustering. Pattern Recognition, 2022.

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
        idx_hdset=round(label)==i;    %consider the case of extension
        match=cmatch(idx_hdset,:);
        match_c(i)={match};
    end
    
    %show the matches
    disp_match_multi(match_c,img1,img2,P1,P2,1);

end


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