DECLARE
   a_    VARCHAR2(2000);
   from_ NUMBER := 1017;
   to_   NUMBER := 2088;

   CURSOR get_items IS
      SELECT t.rowid, to_char(rowversion,'YYYYMMDDHH24MISS') objversion, item_id
      FROM media_item_tab t
      WHERE t.item_id BETWEEN from_ AND to_;
BEGIN      
   FOR rec_ IN get_items LOOP
      BEGIN
         MEDIA_ITEM_API.REMOVE__( a_ , rec_.rowid , rec_.objversion , 'DO' );     
         COMMIT;
      EXCEPTION
         WHEN OTHERS THEN
            dbms_output.put_line(rec_.item_id);
            NULL;
      END;
   END LOOP;
END;

