  procedure Format_Text(text_ in varchar2) is
    dummy_ varchar2(2000);
  begin
    dummy_ := replace(text_, chr(30), ';');
    dummy_ := replace(dummy_, chr(31), '=');
    dbms_output.put_line(dummy_);
  end Format_Text;