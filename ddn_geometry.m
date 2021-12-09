function [ix_lid,ix_lide,ix_uide,ix_uid,ix_uod,ix_uode,ix_lode,ix_lod,...
    ixs_lcore,ixs_rcore,nyseps,nx,ny] = ddn_geometry(baserun,DNGEO)

switch DNGEO
    case 'CDN'
        Nnpr = 5;
    case 'DDN'
        Nnpr = 6;
end
npr = zeros(Nnpr,1);
Nnptseg = 6;%same for DDN and CDN
nptseg = zeros(Nnptseg,1);

%Find number of grid points in each segment of separatrices i: nptseg(i)
%--> poloidal indices
fid=fopen([baserun,'carre.dat']);
tline = fgetl(fid);
for i=1:Nnptseg
    s = num2str(i);
    str = ['nptseg(' s];
    while (~strncmp(tline,sprintf(str),8))
        tline = fgetl(fid);
    end
    nptseg(i) = str2num(tline(16:end))-1;
end

%Find number of surfaces in each region i: npr(i) --> radial indices
for i=1:Nnpr
    s = num2str(i);
    str = ['npr(' s];
    while (~strncmp(tline,sprintf(str),5))
        tline = fgetl(fid);
    end
    npr(i) = str2num(tline(13:end))-1;
end
fclose(fid);

% Poloidal indices
%|G| nptseg(6) | nptseg(4) | nptseg(2) |G|G| npseg(1) | nptseg(3) | nptseg(5) |G|
ix_lid = 2;
ix_lide = 1+nptseg(6);
ix_uide = ix_lide+nptseg(4)+1;
ix_uid = ix_uide+nptseg(2)-1;
ix_uod = ix_uid + 3;
ix_uode = ix_uod + nptseg(1)-1;
ix_lode = ix_uode + nptseg(3)+1;
ix_lod = ix_lode + nptseg(5)-1;
nx = ix_lod + 1;
ixs_lcore = (ix_lide+1:ix_uide-1); ixs_rcore = (ix_uode+1:ix_lode-1);
%
% ix_imp = ncread(balfile,'jxi')+1+2;
% ix_omp = ncread(balfile,'jxa')+1+2;

% Radial indices
% CORE|G| npr(5) | npr(1) |G|SOL
% iy_S = 2;
switch DNGEO
    case 'CDN'
        iy_N = 1+npr(5)+npr(1);
%         iy_isep = 1+npr(5)+1;%(=ncread(balfile,'jsep')+1+2);1st SOL ring
%         iy_osep = iy_isep;
    case 'DDN'
        iy_N = 1+npr(6)+npr(1)+npr(2);
%         iy_isep = 1+npr(6)+1;%(=ncread(balfile,'jsep')+1+2);1st SOL ring
%         iy_osep = 1+npr(6)+npr(1)+1;
end
ny = iy_N + 1;

nyseps = npr(1);%number of surfaces between separatrices
end