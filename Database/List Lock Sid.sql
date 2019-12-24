SELECT s.sid, s.serial#, session_id,lock_type,
mode_held,
mode_requested,
blocking_others,
lock_id1
FROM dba_lock l, gv$session s
where (l.session_id = s.sid
and lock_type
NOT IN ('Media Recovery', 'Redo Thread'))
/

alter system kill session 'sid,serial#'
/