%% Read baserun/carre.dat to get geometrical data 
baserun = '/lustre/home/rtatsumi/solps-iter/runs/stepv10_ddn/drsep0mm/ArDpuff/baserun/'; %v10
DNGEO = 'CDN';%'DDN'
[ix_lid,ix_lide,ix_uide,ix_uid,ix_uod,ix_uode,ix_lode,ix_lod,...
    ixs_lcore,ixs_rcore,nyseps] = ddn_geometry(baserun,DNGEO);
divs = [ix_lid ix_uid ix_uod ix_lod];%polodal indices for the divertor targets
divents = [ix_lide ix_uide ix_uode ix_lode];%polodal indices for the divertor entrances
solring = 2;%which SOL ring to analyse
DIVERTOR = 'upper-inner';%which divertor to analyse (currently only upper)

switch DIVERTOR
    case 'upper-inner'
        idiv = 2;%inner-upper%3:outer-upper
        nxdiv = ix_uid-ix_uide;
        ixrange = [ix_uid-1:-1:ix_uide+1];
        ixdiv = ix_uid;
    case 'upper-outer'
        idiv = 3;%inner-upper%3:outer-upper
        nxdiv = ix_uode-ix_uod;
        ixrange = [ix_uod+1:ix_uode-1];
        ixdiv = ix_uod;
end

%% Read balance.nc for B data
filename = '/lustre/home/rtatsumi/solps-iter/runs/stepv10_ddn/drsep0mm/ArDpuff/P100nu1p75e22Ar3e12Dpuff1e22/';
balfile = [filename 'balance.nc'];
iy_isep = ncread(balfile,'jsep')+1+2;
iy = iy_isep + (solring -1);
comuse = get_comuse(balfile);

nxabs = 1000;%Resolution

[Bix,Lpara,CoverCt,dspolx_cum_use,BX,CoverC0,Bp] = calc_DLS(comuse,divents,divs,idiv,iy,nxabs);
%       Tu_DLS = (BoverBuds(2:end).*qpllu.*3.5./kappa0).^(2.0/7.0);
CoverCt_DLS = CoverCt(1:nxabs);
dspolx_cum_use_DLS = dspolx_cum_use(1:nxabs);

figure('windowstyle','docked');
hold on;
xlabel('Normalised C ');
ylabel('poloidal distance from the target (m) : inner-upper');
plot(CoverCt(1:nxabs),dspolx_cum_use(1:nxabs),'.-')
plot(CoverC0(1:nxabs)./CoverC0(1),dspolx_cum_use(1:nxabs),'.-')
% plot(C_sf_01,dpol,'o-')
legend('C/Ct','C/C_0')%They should be same, just for check

figure('windowstyle','docked');
hold on;
xlabel('poloidal distance from the target (m) : inner-upper');
ylabel('magnetic fields (T)');
% ylim([0 6])
plot(dspolx_cum_use(1:nxabs),Bix(1:nxabs),'-')
plot(dspolx_cum_use(1:nxabs),abs(Bp(1:nxabs)),'--')
legend('|B|','|Bp|')
