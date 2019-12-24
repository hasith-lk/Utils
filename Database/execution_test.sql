spool 'D:\Temp\Log\output.log'

set serveroutput on

exec Installation_SYS.Log_Time_Stamp_Setup('TRUE');
/

exec Installation_SYS.Log_Detail_Time_Stamp('COST','1200.upg','Timestamp_A');
PROMPT Adding columns STD_ORDER_SIZE, USE_LATEST_VALID_STRUCT_REV, USE_LATEST_VALID_ROUT_REV, TEXT_ID$
PROMPT and ROUTING_REVISION to the table PART_COST_TAB.
DECLARE
   columns_    Database_SYS.ColumnTabType;
   table_name_ VARCHAR2(30) := 'PART_COST_TAB';
BEGIN
   Database_SYS.Set_Table_Column(columns_, 'STD_ORDER_SIZE'             , 'NUMBER');
   Database_SYS.Set_Table_Column(columns_, 'USE_LATEST_VALID_STRUCT_REV', 'VARCHAR2(5)');
   Database_SYS.Set_Table_Column(columns_, 'USE_LATEST_VALID_ROUT_REV'  , 'VARCHAR2(5)');
   Database_SYS.Set_Table_Column(columns_, 'TEXT_ID7$'                   , 'VARCHAR2(50)', 'N', 'SYS_GUID()');
   Database_SYS.Set_Table_Column(columns_, 'ROUTING_REVISION'           , 'VARCHAR2(4)' , 'Y');
   Database_SYS.Alter_Table(table_name_, columns_);
END;
/
/*
exec Installation_SYS.Log_Detail_Time_Stamp('COST','1200.upg','Timestamp_B');
PROMPT Updating the values of columns STD_ORDER_SIZE, USE_LATEST_VALID_STRUCT_REV and USE_LATEST_VALID_ROUT_REV in PART_COST_TAB.
PROMPT Making columns STD_ORDER_SIZE, USE_LATEST_VALID_STRUCT_REV and USE_LATEST_VALID_ROUT_REV in PART_COST_TAB to NOT NULL.
DECLARE
   columns_ Database_SYS.ColumnTabType;
   table_name_    VARCHAR2(30) := 'PART_COST_TAB';
BEGIN

   /*
   UPDATE part_cost_tab
      SET text_id6$ = ROWID;
   COMMIT;*/
   

   Database_SYS.Set_Table_Column(columns_, 'TEXT_ID6$'                   , 'VARCHAR2(50)', 'N');
   Database_SYS.Alter_Table(table_name_, columns_);
END;
/
*/
exec Installation_SYS.Log_Detail_Time_Stamp('COST','1200.upg','Timestamp_C');
PROMPT Adding columns STD_ORDER_SIZE, USE_LATEST_VALID_STRUCT_REV, USE_LATEST_VALID_ROUT_REV, TEXT_ID$
PROMPT and ROUTING_REVISION to the table PART_COST_TAB.
DECLARE
   columns_    Database_SYS.ColumnTabType;
   table_name_ VARCHAR2(30) := 'PART_COST_TAB';
BEGIN
   Database_SYS.Set_Table_Column(columns_, 'STD_ORDER_SIZE'             , 'NUMBER');
   Database_SYS.Set_Table_Column(columns_, 'USE_LATEST_VALID_STRUCT_REV', 'VARCHAR2(5)');
   Database_SYS.Set_Table_Column(columns_, 'USE_LATEST_VALID_ROUT_REV'  , 'VARCHAR2(5)');
   -- Database_SYS.Set_Table_Column(columns_, 'TEXT_ID6$'                   , 'VARCHAR2(50)', 'N', 'SYS_GUID()');
   Database_SYS.Set_Table_Column(columns_, 'ROUTING_REVISION'           , 'VARCHAR2(4)' , 'Y');
   Database_SYS.Alter_Table(table_name_, columns_);
END;
/

exec Installation_SYS.Log_Detail_Time_Stamp('COST','1200.upg','Timestamp_D');
PROMPT end test
/

set serveroutput off

spool off
