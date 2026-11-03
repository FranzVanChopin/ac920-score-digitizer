`timescale 1ns / 1ps
// Board S2/S3 have external 4.7k pull-ups to 3.3V. Synchronize into
// AXI control clock; software provides independent 30ms debouncers.
module camera_key_sync (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET resetn" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire resetn,
    input wire key_s2_n,
    input wire key_s3_n,
    output wire [1:0] keys_n
);
    (* ASYNC_REG = "TRUE" *) reg [1:0] meta = 2'b11;
    (* ASYNC_REG = "TRUE" *) reg [1:0] synced = 2'b11;
    always @(posedge clk) begin
        if (!resetn) begin meta <= 2'b11; synced <= 2'b11; end
        else begin meta <= {key_s3_n,key_s2_n}; synced <= meta; end
    end
    assign keys_n = synced;
endmodule
