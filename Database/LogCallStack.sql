-- *****************************************************
--  This Script will log call stack in table call CALL_STACK_LOG.
--  
--  To use the script, go to the method where want to log and
--  add following code line. CALL_STACK_LOG_API.Log_Caller('run_id_');
--
--  Add some id for run_id_ to track down later.
--  
--  Make sure table is truncated before a new run. or change the run_id_
--
-- *****************************************************
--  
-- DROP TABLE CALL_STACK_LOG;
/

CREATE TABLE CALL_STACK_LOG
(
  run_id        VARCHAR2(200),
  caller        VARCHAR2(200),
  receiver      VARCHAR2(200),
  line_no       NUMBER,
  count         NUMBER,
  timestamp     TIMESTAMP
)
/

CREATE OR REPLACE PACKAGE CALL_STACK_LOG_API IS

  PROCEDURE Log_Caller(run_id_ VARCHAR2);

END CALL_STACK_LOG_API;
/


CREATE OR REPLACE PACKAGE BODY CALL_STACK_LOG_API IS

counter_ NUMBER := 1;

PROCEDURE Log_Caller(
   run_id_ VARCHAR2)
IS PRAGMA autonomous_transaction;
   stack_         VARCHAR2(32000);
   depth_         PLS_INTEGER := Utl_Call_Stack.Dynamic_Depth();
   prev_method_   VARCHAR2(200);
   method_        VARCHAR2(200);
   prev_line_     NUMBER;
BEGIN
   FOR i_ IN REVERSE 2 .. depth_ LOOP
     -- dbms_output.put_line(i_||' - '||UTL_Call_Stack.Concatenate_Subprogram(UTL_Call_Stack.Subprogram(i_)) ||' -> '||To_Char(UTL_Call_Stack.Unit_Line(i_)));
     method_ := UTL_Call_Stack.Concatenate_Subprogram(UTL_Call_Stack.Subprogram(i_));
     INSERT INTO CALL_STACK_LOG (run_id, 
                                 caller, 
                                 receiver, 
                                 line_no,
                                 count,
                                 timestamp)
         VALUES (run_id_, 
                 prev_method_, 
                 method_ , 
                 prev_line_,
                 counter_,
                 SYSTIMESTAMP);
     
     prev_method_ := method_;
     prev_line_ := UTL_Call_Stack.Unit_Line(i_);
     --stack_ := stack_ || UTL_Call_Stack.Concatenate_Subprogram(UTL_Call_Stack.Subprogram(i_)) || ' at line ' || To_Char(UTL_Call_Stack.Unit_Line(i_)) || chr(10);
     counter_ := counter_ + 1;
   END LOOP;
      COMMIT;
END Log_Caller ;

END CALL_STACK_LOG_API;
/
