-- Create table
create table TEST_LOG
(
  timestamp DATE not null,
  log       VARCHAR2(2000),
  log_num   NUMBER
)
tablespace IFSAPP_DATA
  pctfree 0
  initrans 1
  maxtrans 255
  storage
  (
    initial 64K
    next 1M
    minextents 1
    maxextents unlimited
  );


CREATE OR REPLACE PACKAGE Debug_Log_API IS

procedure LogDebug(id_ in VARCHAR2, message_ IN VARCHAR2, num_ in NUMBER);

procedure PrintDebug(id_ in VARCHAR2);

end Debug_Log_API;
/

CREATE OR REPLACE PACKAGE BODY Debug_Log_API IS

procedure LogDebug(id_ in VARCHAR2, message_ IN VARCHAR2, num_ in NUMBER)
is
PRAGMA AUTONOMOUS_TRANSACTION;
begin
  insert into TEST_LOG (timestamp,log,log_num) values (sysdate, message_, num_);
  commit;
end LogDebug;

procedure PrintDebug(id_ IN VARCHAR2)
is
   cursor get_comp is
      select *
        from TEST_LOG;

begin

  dbms_output.put_line('== '|| id_ ||' ==');

  for rec_ in get_comp  loop
    /*
     dbms_output.put_line('parent lot b '|| rec_.lot_batch_no);
     dbms_output.put_line('Component '|| rec_.component_part_no);
     dbms_output.put_line('Component Ser '|| rec_.component_serial_no);
     dbms_output.put_line('Component Lot '|| rec_.component_lot_batch_no);
     dbms_output.put_line('Component Qty '|| rec_.component_quantity);
     dbms_output.put_line('Component Removed '|| rec_.comp_qty_removed);
     dbms_output.put_line('Component Presen '|| rec_.component_presence);*/
/*
     Trace_SYS.Message('parent lot b '|| rec_.lot_batch_no);
     Trace_SYS.Message('Component '|| rec_.component_part_no);
     Trace_SYS.Message('Component Ser '|| rec_.component_serial_no);
     Trace_SYS.Message('Component Lot '|| rec_.component_lot_batch_no);
     Trace_SYS.Message('Component Qty '|| rec_.component_quantity);
     Trace_SYS.Message('Component Removed '|| rec_.comp_qty_removed);
     Trace_SYS.Message('Component Presen '|| rec_.component_presence);*/
     Trace_SYS.Message('------------------------------------------');

  end loop;

end PrintDebug;

end Debug_Log_API;
/