function [Bix,L,CoverCt,dspolx_cum_use,BX,CoverC0,Bp] = calc_DLS(comuse,divents,divs,idiv,iy,nxabs)

ix_up = divs(idiv);ix_down=divents(idiv); %UP is target, DOWN is X-point!!!!
hx = comuse.hx; % hx
B = comuse.bb; % Mag. field

if ix_down > ix_up
    bbix = B(ix_up:ix_down,iy,:);
    hix = hx(ix_up:ix_down,iy);
    Bix_data = B(ix_up:ix_down,iy,4);
    Bp_data = B(ix_up:ix_down,iy,1);
else
    bbix = B(ix_up:-1:ix_down,iy,:);
    hix = hx(ix_up:-1:ix_down,iy);
    Bix_data = B(ix_up:-1:ix_down,iy,4);
    Bp_data = B(ix_up:-1:ix_down,iy,1);
end

%obtain L_{||} with the original grid
nxabs_data = abs(ix_up-ix_down)+1;
dsparx_cum = zeros(nxabs_data,1);
dsparx_cum(1)=0.5.*hix(1)*abs(bbix(1,4)/bbix(1,1));
for ix=2:nxabs_data
    dsparx_cum(ix)=dsparx_cum(ix-1)+0.5.*hix(ix-1)*abs(bbix(ix-1,4)/bbix(ix-1,1))+0.5.*hix(ix)*abs(bbix(ix,4)/bbix(ix,1)); %FIX OM 18/4/2019 from bb(ix-1,iy,3)->bb(ix-1,iy,4)
end
ix = nxabs_data;
L = dsparx_cum(nxabs_data) + 0.5.*hix(ix)*abs(bbix(ix,4)/bbix(ix,1));

%obtain the poloidal coordinate with the original grid
dspolx_cum = zeros(nxabs_data,1);
dspolx_cum(1)=0.5.*hix(1);
for ix=2:nxabs_data
    dspolx_cum(ix)=dspolx_cum(ix-1)+0.5.*hix(ix-1)+0.5.*hix(ix);
end
% ix = nxabs_data;
% Lp = dspolx_cum(nxabs_data) + 0.5.*hix(ix);

%Obtaing Bt and BX for each side of boundary
Bt = interp1(dsparx_cum,Bix_data,0,'linear','extrap');
BX = interp1(dsparx_cum,Bix_data,L,'linear','extrap');
Bp_t = interp1(dsparx_cum,Bp_data,0,'linear','extrap');
Bp_X = interp1(dsparx_cum,Bp_data,L,'linear','extrap');
%Refine Bix
dxpara = L./double(nxabs);
dsparx_cum_use = zeros(nxabs+1,1);
for ix = 1:nxabs
    dsparx_cum_use(ix) = dxpara.*double(ix-1);
end
dsparx_cum_use(nxabs+1) = L;
Bix  = zeros(nxabs+1,1);
Bp = zeros(nxabs+1,1);
Bix(1) = Bt;
Bp(1) = Bp_t;
for ix = 2:nxabs
    if dsparx_cum_use(ix) < dsparx_cum(1) || dsparx_cum_use(ix) > dsparx_cum(end)
        Bix(ix) = interp1(dsparx_cum(:),Bix_data(:),dsparx_cum_use(ix),'linear','extrap');
Bp(ix) = interp1(dsparx_cum(:),Bp_data(:),dsparx_cum_use(ix),'linear','extrap');
    else
        Bix(ix) = interp1(dsparx_cum(:),Bix_data(:),dsparx_cum_use(ix),'linear');
 Bp(ix) = interp1(dsparx_cum(:),Bp_data(:),dsparx_cum_use(ix),'linear');
    end
end
Bix(nxabs+1) = BX;
Bp(nxabs+1) = Bp_X;

%Refine dspolx_cum
dspolx_cum_use = zeros(nxabs,1);
dspolx_cum_use(1) = 0;
% dspolx_cum_use(nxabs+1) = Lp;
for ix = 2:nxabs
    if dsparx_cum_use(ix) < dsparx_cum(1) || dsparx_cum_use(ix) > dsparx_cum(end)
        dspolx_cum_use(ix) = interp1(dsparx_cum(:),dspolx_cum(:),dsparx_cum_use(ix),'linear','extrap');
    else
        dspolx_cum_use(ix) = interp1(dsparx_cum(:),dspolx_cum(:),dsparx_cum_use(ix),'linear');
    end
end

%Int_f^u Bds
IntBds = zeros(nxabs,1);
for ix = 1:nxabs
    IntBds(ix) = sum(Bix(ix:end).*dsparx_cum_use(ix:end));
end
CoverCt= zeros(nxabs,1);
CoverC0= zeros(nxabs,1);
for ix = 1:nxabs
    CoverCt(ix) = Bix(ix)./Bt.*(IntBds(ix)./IntBds(1)).^(-2/7);
    CoverC0(ix) = Bix(ix)./BX.*(IntBds(ix)./BX).^(-2/7);
end

end