-- Find large lob tables
SELECT *--sum(SIZE_GB) 
FROM (
  SELECT 
     t2.table_name, t2.column_name, SEGMENT_TYPE, t1.SEGMENT_NAME, BYTES/1024/1024/1024 SIZE_GB 
  FROM 
    user_SEGMENTS t1, user_lobs t2
 WHERE
      SEGMENT_TYPE ='LOBSEGMENT'
and t1.segment_name = t2.SEGMENT_NAME
  ORDER BY 
    BYTES/1024/1024  DESC ) WHERE ROWNUM <= 12;
/

-- Show available free space
select ddf.file_name, ddf.tablespace_name, sum(dfs.bytes)/1024/1024 free_space_MB
from dba_data_files ddf, dba_free_space dfs
where ddf.file_id = dfs.file_id
and ddf.tablespace_name like 'IFSAPP_LOB'
group by ddf.file_name,ddf.tablespace_name
/

-- Show large LOB records
select t.doc_no, dbms_lob.getlength(&column_name)/1024/1024 from &Table_Name t
order by dbms_lob.getlength(&column_name) desc;
/

-- Remove unused data
/

-- Shrink table
alter table EDM_FILE_STORAGE_tab enable row movement;
/
alter table EDM_FILE_STORAGE_tab shrink space;
/

alter table EDM_FILE_STORAGE_tab modify lob (file_data) (freepools 1);
alter table EDM_FILE_STORAGE_tab modify lob (file_data) (shrink space cascade);
alter table EDM_FILE_STORAGE_tab modify lob (file_data) (pctversion 0);
alter table EDM_FILE_STORAGE_tab modify lob (file_data) (retention);
/

-- resize table space
alter database datafile 'C:\IFS\ORADB\SB\IFSAPP_LOB.DBF' resize 15G;