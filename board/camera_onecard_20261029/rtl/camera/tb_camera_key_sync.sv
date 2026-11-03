`timescale 1ns/1ps
module tb_camera_key_sync;
reg clk=0,resetn=0,key_s2_n=1,key_s3_n=1;
wire [1:0] keys_n;
always #5 clk=~clk;
camera_key_sync dut(.*);
task tick; begin @(posedge clk); #1; end endtask
initial begin
 tick(); if(keys_n!==2'b11) $fatal(1,"reset levels");
 resetn=1;#2;key_s2_n=0;
 tick(); if(keys_n!==2'b11) $fatal(1,"key bypassed first sync stage");
 tick(); if(keys_n!==2'b10) $fatal(1,"S2 mapping/latency");
 #2;key_s3_n=0;tick();tick(); if(keys_n!==2'b00) $fatal(1,"S3 mapping");
 #2;key_s2_n=1;tick();tick(); if(keys_n!==2'b01) $fatal(1,"independent release");
 resetn=0;tick(); if(keys_n!==2'b11) $fatal(1,"reset must release both inputs");
 resetn=1;tick();tick(); if(keys_n!==2'b01) $fatal(1,"held key after reset");
 $display("PASS camera_key_sync: two-stage latency, polarity, mapping and reset");$finish;
end
endmodule
