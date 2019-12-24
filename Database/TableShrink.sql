-- Find largest segments
select t.segment_name, t.BYTES/1024/1024/1024 GB, t.segment_type from SYS.USER_SEGMENTS t order by bytes desc;

-- Show available free space
select ddf.file_name, ddf.tablespace_name, sum(dfs.bytes)/1024/1024 free_space_MB
from dba_data_files ddf, dba_free_space dfs
where ddf.file_id = dfs.file_id
and ddf.tablespace_name like 'IFSAPP_DATA'
group by ddf.file_name,ddf.tablespace_name
/

-- truncate data
truncate table DR$DOC_ISSUE_CTIX3$I;

-- enable shrink
alter table DR$DOC_ISSUE_CTIX3$I enable row movement;
alter table DR$DOC_ISSUE_CTIX3$I shrink space;

-- Resize data file RUN AS SYS
alter database datafile 'C:\IFS\ORADB\SB\IFSAPP_DATA.DBF' resize 20G;