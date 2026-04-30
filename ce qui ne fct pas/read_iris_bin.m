%% Written by J. GANCE
%% V3.0 - 04/11/2019
%% Matlab code to read IRIS Instruments .bin and .pro files
% Select the .bin or .pro file

[filename,path] = uigetfile({'*.bin';'*.pro'},'File selector');
if isequal(filename,0)
    return;
else
    file = [path filename];
    fid=fopen(file, 'rb');
    fseek(fid,  0,'eof');
    TailleTotale=ftell(fid);
    fseek(fid, 0, 'bof');
    
    if(filename(end-2:end) == 'bin')
        
        data.version=fread(fid,1,'ulong');
        data.TypeOfSyscal=fread(fid,1,'uint8');
        
        if((data.TypeOfSyscal == 8 || data.TypeOfSyscal == 9 || data.TypeOfSyscal == 3 || data.TypeOfSyscal == 11 || data.TypeOfSyscal == 4 || data.TypeOfSyscal == 5 || data.TypeOfSyscal == 10) || (data.version >= 2147483650 && (data.TypeOfSyscal == 2 || data.TypeOfSyscal == 7 || data.TypeOfSyscal == 1 || data.TypeOfSyscal == 6 )))
            data.comment=fread(fid,1024,'*char');
            data.comment=data.comment';
        end
        
        if(((data.TypeOfSyscal == 8 || data.TypeOfSyscal == 9 || data.TypeOfSyscal == 3 || data.TypeOfSyscal == 11 || data.TypeOfSyscal == 4 || data.TypeOfSyscal == 5) && data.version == 2147483651) || ((data.TypeOfSyscal == 1 || data.TypeOfSyscal == 6 || data.TypeOfSyscal == 10) && data.version >= 2147483651))
            data.ColeCole=fread(fid,[64000 3],'float32');
        end
        
        if(data.version >= 2147483652) %0x80000004 en  HEXA
            data.CommonFilePath=fread(fid,260,'*char');
            data.NbFiles=fread(fid,1,'ushort');
            for i=1:data.NbFiles
                data.SizeFileName{i}=fread(fid,1,'ushort');
                data.FileNameIabOrVmn{i}=fread(fid,data.SizeFileName{i},'*char');
            end
        end
        
        if(data.TypeOfSyscal == 8 || data.TypeOfSyscal == 9 || data.TypeOfSyscal == 3 || data.TypeOfSyscal == 11 || data.TypeOfSyscal == 4 || data.TypeOfSyscal == 5)
            Position=ftell(fid);
            i=1;
            tic
            h = waitbar(0,'Reading the BIN file...');
            while(Position < TailleTotale)
                waitbar(Position / TailleTotale);
                % Reading electrode array
                data.Measure(i).el_array=fread(fid, 1, 'int16');
                % Reading ???
                data.Measure(i).MoreTMesure=fread(fid, 1, 'short');
                % Reading Injection time
                data.Measure(i).time=fread(fid, 1, 'float32');
                % Reading M_delay
                data.Measure(i).m_dly =fread(fid, 1, 'float32');
                % Reading Kid or not
                data.Measure(i).TypeCpXyz=fread(fid, 1, 'int16');
                if(data.Measure(i).TypeCpXyz == 0)
                    display('These data have been recorded with the Syscal KID and can''t be read');
                end
                % Reading ignored parameter
                data.Measure(i).Q=fread(fid, 1, 'int16');
                % Reading electrode positions
                data.Measure(i).pos=fread(fid, 12, 'float32');
                % Reading PS
                data.Measure(i).Ps=fread(fid, 1, 'float32');
                % Reading Vp
                data.Measure(i).Vp=fread(fid, 1, 'float32');
                % Reading In
                data.Measure(i).In=fread(fid, 1, 'float32');
                % Reading resistivity
                data.Measure(i).rho=fread(fid, 1, 'float32');
                % Reading chargeability
                data.Measure(i).m=fread(fid, 1, 'float32');
                % Reading Q
                data.Measure(i).dev=fread(fid, 1, 'float32');
                % Reading IP Windows duration (Tm)
                data.Measure(i).Tm(1:20)=fread(fid, 20, 'float32');
                % Reading IP Windows values
                data.Measure(i).Mx(1:20)=fread(fid, 20, 'float32');
                
                data.Measure(i).Channel=fread(fid,1,'ubit4');
                data.Measure(i).NbChannel=fread(fid,1,'ubit4');
                data.Measure(i).Overload=fread(fid,1,'ubit1');
                data.Measure(i).ChannelValide=fread(fid,1,'ubit1');
                data.Measure(i).unused=fread(fid,1,'ubit6');
                data.Measure(i).QuadNumber=fread(fid,1,'ubit16');
                data.Measure(i).Name(1:12)=fread(fid,12,'char');
                data.Measure(i).Latitude=fread(fid, 1, 'float32');
                data.Measure(i).Longitude=fread(fid, 1, 'float32');
                data.Measure(i).NbCren=fread(fid, 1, 'float32');
                data.Measure(i).RsChk=fread(fid, 1, 'float32');
                if(data.Measure(i).MoreTMesure == 2)
                    data.Measure(i).TxVab=fread(fid, 1, 'float32');
                    data.Measure(i).TxBat=fread(fid, 1, 'float32');
                    data.Measure(i).RxBat=fread(fid, 1, 'float32');
                    data.Measure(i).Temperature=fread(fid, 1, 'float32');
                elseif(data.Measure(i).MoreTMesure == 3)
                    data.Measure(i).TxVab=fread(fid, 1, 'float32');
                    data.Measure(i).TxBat=fread(fid, 1, 'float32');
                    data.Measure(i).RxBat=fread(fid, 1, 'float32');
                    data.Measure(i).Temperature=fread(fid, 1, 'float32');
                    data.Measure(i).DateTime=fread(fid, 1, 'double');
                    data.Measure(i).DateTime=datenum(data.Measure(i).DateTime + datenum([1899 12 30 00 00 00]));
                end
                if(data.version >= 2147483652) %0x80000004 en  HEXA
                    data.Measure(i).Iabfile=fread(fid,1,'short');
                    data.Measure(i).Vmnfile=fread(fid,1,'short');
                    
                end
                Position=ftell(fid);
                i=i+1;
            end;
            toc;
            fclose(fid);
            close(h);
        else
            disp('Device not managed for the moment');
        end
    elseif(filename(end-2:end) == 'pro')
        data.download_version=fread(fid,1,'int32');
        data.size=fread(fid,1,'int32');
        page = -1;
        i=0;
        while ~feof(fid)
            i = i+1;
            page = fread(fid,1,'int32');
            test = fread(fid,1,'uint32');
            if(isempty(test))
                break;
            end
            data.Measure(i).date = test;
            data.Measure(i).Name= char(fread(fid,12,'char'));
            data.Measure(i).Channel=fread(fid,1,'ubit4');
            data.Measure(i).NbChannel=fread(fid,1,'ubit4');
            data.Measure(i).Overload=fread(fid,1,'ubit1');
            data.Measure(i).ChannelValide=fread(fid,1,'ubit1');
            data.Measure(i).unused=fread(fid,1,'ubit6');
            data.Measure(i).QuadNumber=fread(fid,1,'ubit16');
            data.Measure(i).vrunning = fread(fid,1,'ubit1');
            data.Measure(i).vsigned = fread(fid,1,'ubit1');
            data.Measure(i).normalized = fread(fid,1,'ubit1');
            data.Measure(i).imperial = fread(fid,1,'ubit1');
            data.Measure(i).unused = fread(fid,1,'ubit4');
            data.Measure(i).timeSet = fread(fid,1,'ubit4');
            data.Measure(i).timeMode = fread(fid,1,'ubit4');
            data.Measure(i).type = fread(fid,1,'ubit8');
            data.Measure(i).unused2 = fread(fid,1,'ubit8');
            data.Measure(i).el_array = fread(fid,1,'int32');
            data.Measure(i).time=fread(fid, 1, 'ushort');
            data.Measure(i).vdly=fread(fid, 1, 'ushort');
            data.Measure(i).mdly=fread(fid, 1, 'ushort');
            data.Measure(i).tm(1:20)=fread(fid, 20, 'ushort');
            data.Measure(i).unused3 =fread(fid, 1,'ushort');
            data.Measure(i).Latitude=fread(fid, 1, 'float32');
            data.Measure(i).Longitude=fread(fid, 1, 'float32');
            data.Measure(i).inrx=fread(fid, 1, 'float32');
            data.Measure(i).pos=fread(fid, 12, 'float32');
            data.Measure(i).mov=fread(fid, 3, 'float32');
            data.Measure(i).rho=fread(fid, 1, 'float32');
            data.Measure(i).dev=fread(fid, 1, 'float32');
            data.Measure(i).NbCren=fread(fid, 1, 'float32');
            data.Measure(i).RsChk=fread(fid, 1, 'float32');
            data.Measure(i).TxVab=fread(fid, 1, 'float32');
            data.Measure(i).TxBat=fread(fid, 1, 'float32');
            data.Measure(i).RxBat=fread(fid, 1, 'float32');
            data.Measure(i).Temperature=fread(fid, 1, 'float32');
            data.Measure(i).Ps=fread(fid, 1, 'float32');
            data.Measure(i).Vp=fread(fid, 1, 'float32');
            data.Measure(i).In=fread(fid, 1, 'float32');
            data.Measure(i).m=fread(fid, 1, 'float32');
            data.Measure(i).Mx(1:20)=fread(fid, 20, 'float32');
            rcrc = fread(fid,1,'uint32');
            if((page == -1))
                return;
            end
        end
    end
end