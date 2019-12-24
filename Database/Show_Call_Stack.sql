CREATE OR REPLACE PACKAGE Show_Call_Stack IS

FUNCTION Show RETURN VARCHAR2;

END Show_Call_Stack;

/

CREATE OR REPLACE PACKAGE BODY Show_Call_Stack IS
FUNCTION Show RETURN VARCHAR2
IS
   stack_ VARCHAR2(32000);
   depth_ PLS_INTEGER := Utl_Call_Stack.Dynamic_Depth();
BEGIN
   FOR i_ IN REVERSE 1 .. depth_ LOOP
      stack_ := stack_ || UTL_Call_Stack.Concatenate_Subprogram(UTL_Call_Stack.Subprogram(i_)) || ' at line ' || To_Char(UTL_Call_Stack.Unit_Line(i_)) || chr(10)|| chr(13);
   END LOOP;
   RETURN stack_;
END show;
END Show_Call_Stack;

/

prompt Deploy end ...