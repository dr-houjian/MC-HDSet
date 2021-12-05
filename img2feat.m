%This is detect features and descriptors.

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