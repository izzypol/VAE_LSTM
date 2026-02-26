function data = fast_read_iab_or_vmn_file(file)

%% Function written by Jacques Deparis - BRGM 2018
% V1.0 28-02-2018 -> Reading number of first sample with type ubit24
% For quick reading V and I-FullWaver data files


%% read data
format compact; format longG;
fid = fopen(file,'rb');
data = struct();
tmp = fread(fid,'*char'); % première lecture du fichier en caractères
Ltmp = length(tmp);
PC(1) = 0;
j = 0;
for i = 1:528:Ltmp-1
    if tmp(i) == 'C'
        j=j+1;  PC(j)=i;
    end
end
Lpr = length(PC);
Lpc = Lpr+1;
PC(Lpc) = Ltmp; %     Lpc = length(PC)
PR(1:Lpr) = 0;  %     Lpr = length(PR)

%%read var
for i = 1:Lpr
    for k = PC(i):528:PC(i+1)
        if tmp(k)=='R'
            PR(i)=k;
        end
    end
end
PR = PR+527;
fseek(fid,0,'bof');    tmf = fread(fid,'float'); % rembobinage du fichier puis lecture en réels (4 octets)
fseek(fid,0,'bof');    tmi = fread(fid,'int');   % rembobinage du fichier puis lecture en entiers signés (4 octets)
fseek(fid,0,'bof');    tmu4 = int8(fread(fid,'ubit4'));  % lecture en 4bit (4/8 octets) 
fseek(fid,0,'bof');    tmui = fread(fid,'uint'); % rembobinage du fichier puis lecture en entiers non signés (4 octets)

for i = 1:Lpr
    if PR(i) < PC(i)
        PR(i)=PC(i+1)-1;
        warning('Error when loading file');
    end
    tmp2 = tmp(PC(i):PR(i));
    data(i).Ctrl = tmp2(1);
    data(i).typeacqu = tmp2(2:12)';
    if Lpr==1
        clear tmp
    end
    %%%%% floats %%%%%
    PC2(1:Lpr) = PC(1:Lpr); %% on abandonne le dernier élément de PC (donc on retrouve le tmp du départ ?)
    PC2(2:end)=(PC2(2:end)-1)/4 + 1;
    PR2 = PR/4;
    tmp2 = tmf(PC2(i):PR2(i)); %% on travaille sur la lecture en rééls
    data(i).Geo.x(1:13)=tmp2(5:2:29);
    data(i).Geo.y(1:13)=tmp2(6:2:30);
    data(i).lat = tmp2(31);
    data(i).lon = tmp2(32);
    data(i).InRx = tmp2(33);
    data(i).GmtOffset = tmp2(36);
    data(i).ZGps = tmp2(37);
    data(i).MoveX = tmp2(38);
    data(i).MoveY = tmp2(39);
    %%%%% entiers signés %%%%%
    tmp2 = tmi(PC2(i):PR2(i)); %% on travaille sur la lecture en entiers signés
    data(i).NbChannel = tmp2(34);
    data(i).MaxInputVoltage = tmp2(35);
    data(i).unused = tmp2(40:51);
    if Lpr==1
           clear tmi
    end
    %%%%% entiers non signés %%%%%
    tmp2 = tmui(PC2(i):PR2(i)); %% on travaille sur la lecture en entiers non signés
    if size(tmp2,1) < 136
        data(i).date = tmp2(4);
        data(i).dateor = tmp2(4);
    else
        data(i).date = tmp2(4:132:136)/86400 + 483 + 1989*365; %% date exprimée en jours depuis la naissance du Christ
        data(i).dateor = tmp2(4:132:136); %% date exprimée en secondes depuis le 1er janvier 1990 à 00h00'0"
    end
    data(i).datestr = datestr(data(i).date); %% date exprimée au format administratif
    if Lpr==1
        %             clear tmui
    end
    ByteNU = floor(512/(4*(1+data(i).NbChannel)));
    ByteNU = 512 - ByteNU*4*(1+data(i).NbChannel);
    PC2(1:length(PC)) = PC(1:length(PC));
    PC2(2:end)=PC(2:end)*2-1;
    PR2 = PR*2;
    tmp2 = tmu4(PC2(i):PR2(i)); %% on travaille sur la lecture en 4bit
    tmp2(1:1056)=[];
    tmp2 = reshape(tmp2,1056,length(tmp2)/1056);
    if ByteNU > 0
        tmp2(end-ByteNU*2+1:end,:)=[];
    end
    tmp2(1:32,:)=[];
    tmp2 = reshape(tmp2,size(tmp2,1)*size(tmp2,2),1);
    Ltmp = 8+8*data(i).NbChannel;
    data(i).PPS_Chan_Nb10ms.NbChannel = tmp2(2:Ltmp:end)';
    data(i).PPS_Chan_Nb10ms.Nb10ms = 0:length(data(i).PPS_Chan_Nb10ms.NbChannel)-1;
    if Lpr==1
        clear tmu4
    end
    
    %%%%% time Value %%%%%
    fseek(fid,PC(i)+544,'bof');  indstart = fread(fid, 1,'ubit24'); % rembobinage du fichier au 545ème octet puis lecture d'un seul élément en 24 bits (3 octets)
% NB: indstart correspond à la variable Nb10ms dans le code C++ DemoRawData/Main d'IRIS (ligne 60)

%%%%% Vp %%%%%
    PC2(1:Lpr) = (PC(1:Lpr)-1)/4+1;
    PR2 = PR/4;
    tmp2 = tmf(PC2(i):PR2(i));
    page = length(tmp2)/132-1;
    tmp2(1:132)=[];
    tmp2 = reshape(tmp2,132,page);
    if ByteNU > 0
        tmp2(end-ByteNU/4+1:end,:)=[];
    end
    tmp2(1:4,:)=[];
    tmp2(1:data(i).NbChannel+1:end,:)=[];
    Ltmp = size(tmp2,1)*size(tmp2,2)/data(i).NbChannel;
    tmp2 = reshape(tmp2,data(i).NbChannel,Ltmp);
    tmp2 = tmp2';
    % recuperation du temps
    data.istart=indstart;
    data(i).Value(1:Ltmp,1) = ((0 : 1 : Ltmp-1)+indstart)/100;
    data(i).Value(:,2:data(i).NbChannel+1)=tmp2;
end
fclose(fid);
end

