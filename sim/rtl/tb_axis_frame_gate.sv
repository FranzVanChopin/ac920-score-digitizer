`timescale 1ns / 1ps
module tb_axis_frame_gate;
    reg clk=0, ctrl_clk=0, resetn=0, ctrl_resetn=0;
    always #2.5 clk=~clk;
    always #5 ctrl_clk=~ctrl_clk;
    reg frame_req_toggle=0, frame_abort=0, preview_mode=0;
    reg reset_request=0;
    wire isp_resetn;
    wire [7:0] control_status;
    wire [7:0] capture_status;
    reg [23:0] s_axis_tdata=0;
    reg s_axis_tvalid=0, s_axis_tuser=0, s_axis_tlast=0;
    wire s_axis_tready;
    wire [23:0] m_axis_tdata;
    wire m_axis_tvalid, m_axis_tuser, m_axis_tlast;
    reg m_axis_tready=1;
    integer beats=0, sofs=0, eols=0;
    integer old_beats;
    reg old_done;
    reg [25:0] held;
    reg stalled=0;
    axis_frame_gate #(.FRAME_WIDTH(8), .FRAME_HEIGHT(4),
        .PREVIEW_WIDTH(4), .PREVIEW_HEIGHT(2)) dut(.*);
    camera_isp_control reset_controller(.clk(clk),.resetn(resetn),
        .ctrl_clk(ctrl_clk),.ctrl_resetn(ctrl_resetn),.reset_request(reset_request),
        .gate_status(capture_status),
        .isp_resetn(isp_resetn),.control_status(control_status));

    always @(posedge clk) begin
        if (resetn && m_axis_tvalid && m_axis_tready) begin
            beats <= beats+1;
            if (m_axis_tuser) sofs <= sofs+1;
            if (m_axis_tlast) eols <= eols+1;
        end
        if (resetn && !frame_abort && stalled) begin
            if (!m_axis_tvalid || {m_axis_tuser,m_axis_tlast,m_axis_tdata} !== held)
                $fatal(1,"AXIS payload changed while stalled");
        end
        stalled <= resetn && m_axis_tvalid && !m_axis_tready && !frame_abort;
        held <= {m_axis_tuser,m_axis_tlast,m_axis_tdata};
    end
    task automatic beat(input integer value, input bit sof, input bit eol);
        @(negedge clk);
        s_axis_tvalid=1; s_axis_tdata=value; s_axis_tuser=sof; s_axis_tlast=eol;
        do @(posedge clk); while (!s_axis_tready);
        @(negedge clk); s_axis_tvalid=0;
    endtask
    task automatic frame(input integer width=8, input integer height=4);
        for(integer y=0;y<height;y++)
            for(integer x=0;x<width;x++) beat(y*width+x,x==0&&y==0,x==width-1);
    endtask
    task automatic settle;
        repeat(8) @(negedge ctrl_clk);
    endtask
    task automatic request_frame;
        frame_req_toggle=~frame_req_toggle;
        settle();
        if (!capture_status[3]) $fatal(1,"Request not armed");
    endtask
    initial begin
        repeat(8) @(negedge clk); resetn=1; ctrl_resetn=1;
        frame(); settle();
        if(beats!=0) $fatal(1,"Unrequested frame leaked");
        request_frame();
        // Ignore a tail of the frame already in flight when the request arrived.
        for(integer x=0;x<8;x++) beat(100+x,0,x==7);
        if(beats!=0) $fatal(1,"Started in the middle of a frame");
        fork
            frame();
            begin
                repeat(12) @(negedge clk); m_axis_tready=0;
                repeat(9) @(negedge clk); m_axis_tready=1;
            end
        join
        settle();
        if(beats!=32 || sofs!=1 || eols!=4 || capture_status!=8'h01)
            $fatal(1,"Complete frame mismatch: %d %d %d status=%h",beats,sofs,eols,capture_status);
        frame(); settle();
        if(beats!=32) $fatal(1,"Following frame leaked");
        request_frame(); frame(); settle();
        if(beats!=64 || capture_status!=0) $fatal(1,"Second request failed");
        request_frame(); frame(7,4); settle();
        if(!capture_status[2] || !capture_status[0]) $fatal(1,"Short line not detected");
        request_frame(); frame(9,4); settle();
        if(!capture_status[2] || capture_status[0]) $fatal(1,"Long line not detected");
        request_frame(); frame(8,2); beat(999,1,0); settle();
        if(!capture_status[2] || capture_status[1] || !capture_status[0])
            $fatal(1,"Early SOF not rejected");
        request_frame(); beat(0,1,0); frame_abort=1; settle();
        if(capture_status[3:1]!=3'b010) $fatal(1,"Abort did not quiesce gate");
        old_beats=beats;
        frame(); settle();
        if(beats!=old_beats) $fatal(1,"Frame passed while aborted");
        frame_abort=0; settle(); request_frame(); frame(); settle();
        if(capture_status[3:1]!=0) $fatal(1,"Recovery frame failed");
        // Alternate dimensions repeatedly. Change requested mode AFTER request
        // acceptance, including while busy: the latched dimensions must win.
        for(integer k=0;k<12;k++) begin
            preview_mode=k%2; settle(); old_beats=beats;
            request_frame();
            if(capture_status[4] !== (k%2 != 0)) $fatal(1,"Mode not latched");
            preview_mode=~preview_mode; settle();
            fork
                frame(k%2 ? 4:8,k%2 ? 2:4);
                begin
                    repeat(6) @(negedge clk); m_axis_tready=0;
                    repeat(5) @(negedge clk); m_axis_tready=1;
                end
            join
            settle();
            if(beats-old_beats != (k%2 ? 8:32) || capture_status[3:1]!=0)
                $fatal(1,"Alternating mode frame failed %d status=%h",k,capture_status);
        end
        preview_mode=1; settle(); request_frame(); frame(3,2); settle();
        if(!capture_status[2]) $fatal(1,"Preview short line undetected");
        request_frame(); frame(5,2); settle();
        if(!capture_status[2]) $fatal(1,"Preview long line undetected");
        request_frame(); frame(4,1); beat(999,1,0); settle();
        if(!capture_status[2] || capture_status[1]) $fatal(1,"Preview early SOF undetected");
        request_frame(); beat(0,1,0); frame_abort=1; settle();
        if(capture_status[3:1]!=3'b010) $fatal(1,"Preview abort failed");
        frame_abort=0; settle(); request_frame(); frame(4,2); settle();
        if(capture_status[3:1]!=0) $fatal(1,"Preview abort recovery failed");
        // ISP-only reset must acknowledge in ctrl domain and preserve gate
        // history. Reserved former S3 status remains released.
        old_done=capture_status[0]; reset_request=1; settle();
        if(isp_resetn || !control_status[5]) $fatal(1,"ISP reset assert/ack failed");
        if(capture_status[0]!==old_done) $fatal(1,"ISP reset changed gate done");
        if(!control_status[4]) $fatal(1,"Reserved status must be high");
        reset_request=0; settle();
        if(!isp_resetn || control_status[5] || !control_status[4])
            $fatal(1,"ISP release/reserved status failed");
        if(control_status[6]!==capture_status[4]) $fatal(1,"Accepted mode status mapping failed");
        $display("PASS axis_frame_gate: dual dimensions, mode locked per request, malformed frames, abort recovery, backpressure, ISP reset isolation/ack and reserved status");
        $finish;
    end
    initial begin #200000; $fatal(1,"Test timed out"); end
endmodule
