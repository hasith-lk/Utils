SELECT owner,
segment_name,
segment_type,
tablespace_name,
bytes/1048576 MB,
initial_extent,
next_extent,
extents,
pct_increase
FROM
DBA_SEGMENTS
WHERE
OWNER = 'IFSAPP' AND
SEGMENT_TYPE = 'TABLE'
order by bytes desc
/