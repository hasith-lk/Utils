CREATE OR REPLACE PACKAGE MRP_Debug IS
   PROCEDURE Print_String(msg_ IN VARCHAR2);
   PROCEDURE Print_Part_Supp_Demand(contract_ IN VARCHAR2, part_no_ IN VARCHAR2);
   PROCEDURE Print_Part_Event(contract_ IN VARCHAR2, part_no_ IN VARCHAR2);
   PROCEDURE Print_Part_Event(pref_ VARCHAR2, contract_ IN VARCHAR2, part_no_ IN VARCHAR2);
END MRP_Debug;
/
   
CREATE OR REPLACE PACKAGE BODY MRP_Debug IS

PROCEDURE Print_String(msg_ IN VARCHAR2)
   IS
BEGIN
   dbms_output.put_line(msg_);
END Print_String;

PROCEDURE Print_Part_Supp_Demand(contract_ IN VARCHAR2, part_no_ IN VARCHAR2) IS
   CURSOR get_demand IS
   SELECT *
      FROM mrp_part_supply_demand_tab
      WHERE contract = contract_
      AND part_no = part_no;
BEGIN
      FOR rec_ IN get_demand LOOP
         Print_String('Part Supp Demand : Part -> '|| rec_.part_no || '; Site -> '||rec_.contract ||'; Supply -> '||rec_.supply_qty ||'; Demand -> '||rec_.demand_qty
                      || 'Order det -> '|| rec_.order_no||'-'||rec_.line_no||'-'||rec_.release_no||'-'||rec_.line_item_no||'; MRP SOURCE -> '|| rec_.mrp_source);
      END LOOP;
END Print_Part_Supp_Demand;

PROCEDURE Print_Part_Event(pref_ VARCHAR2, contract_ IN VARCHAR2, part_no_ IN VARCHAR2) IS
   CURSOR get_demand IS
   SELECT *
      FROM mrp_part_event_tab
      WHERE contract = contract_
      AND part_no = part_no_;
BEGIN
      FOR rec_ IN get_demand LOOP
         Print_String(pref_ || ' Part event : Part -> '|| rec_.part_no || '; Site -> '||rec_.contract ||'; Event Date -> '|| rec_.event_date||
                      '; Gross -> '|| rec_.gross_requirement || '; Net -> '|| rec_.net_requirement ||'; MRP Proj Onhand -> ' || rec_.mrp_projected_onhand||
                      '; By prod Rcpt -> '|| rec_.mrp_by_prod_receipt);
      END LOOP;
END Print_Part_Event;

PROCEDURE Print_Part_Event(contract_ IN VARCHAR2, part_no_ IN VARCHAR2) IS
BEGIN
   Print_Part_Event('', contract_, part_no_);
END Print_Part_Event;

END MRP_Debug;
/
