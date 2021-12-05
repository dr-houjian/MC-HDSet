%This is to extract matches with hdset.

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
                    label(j)=0;              %matches in large-sequence-number clusters are discarded
                end
                if cmatch(i,2)==cmatch(j,2)
                    label(j)=0;
                end
            end
        end
    end

end
