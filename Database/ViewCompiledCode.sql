BEGIN
  DBMS_PREPROCESSOR.print_post_processed_source (
    object_type => 'PROCEDURE',
    schema_name => 'TEST',
    object_name => 'DEBUG');
END;
/