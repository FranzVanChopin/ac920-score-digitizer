`timescale 1ns/1ps

module tb_axi_score_event_player;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [11:0] awaddr = 0;
    logic awvalid = 0;
    logic awready;
    logic [31:0] wdata = 0;
    logic [3:0] wstrb = 4'hF;
    logic wvalid = 0;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready = 0;
    logic [11:0] araddr = 0;
    logic arvalid = 0;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready = 0;
    logic beep_out;
    logic [31:0] status;
    time first_edge;
    time half_period_ns;

    always #5000 clk = ~clk;

    axi_score_event_player #(.CLK_HZ(100_000)) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .beep_out(beep_out)
    );

    task automatic write_reg(input logic [11:0] address, input logic [31:0] data);
        begin
            @(negedge clk);
            awaddr = address;
            wdata = data;
            awvalid = 1'b1;
            wvalid = 1'b1;
            while (!(awready && wready)) @(negedge clk);
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            while (!bvalid) @(negedge clk);
            bready = 1'b1;
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    task automatic read_reg(input logic [11:0] address, output logic [31:0] data);
        begin
            @(negedge clk);
            araddr = address;
            arvalid = 1'b1;
            while (!arready) @(negedge clk);
            @(negedge clk);
            arvalid = 1'b0;
            while (!rvalid) @(negedge clk);
            data = rdata;
            rready = 1'b1;
            @(negedge clk);
            rready = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        write_reg(6'h04, 32'd69); // A4
        write_reg(6'h08, 32'd3);  // 3 ms
        write_reg(6'h00, 32'd1);  // start

        read_reg(6'h00, status);
        if (!status[1] || status[0])
            $fatal(1, "status did not report the active event");

        wait (beep_out === 1'b1);
        first_edge = $time;
        wait (beep_out === 1'b0);
        half_period_ns = $time - first_edge;
        if (half_period_ns != 1_130_000)
            $fatal(1, "A4 half-period mismatch: %0t ns", half_period_ns);

        wait (dut.done_sticky === 1'b1);
        read_reg(6'h00, status);
        if (!status[0] || !status[3])
            $fatal(1, "status did not report ready/done after playback");
        write_reg(6'h00, 32'd2); // clear sticky done
        read_reg(6'h00, status);
        if (status[3])
            $fatal(1, "done sticky bit did not clear");
        $display("PASS: AXI register writes, status, and tone event");
        $finish;
    end

    initial begin
        #5_000_000;
        $fatal(1, "timeout waiting for AXI playback");
    end
endmodule
