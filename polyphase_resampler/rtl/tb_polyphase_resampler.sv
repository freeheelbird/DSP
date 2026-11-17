// Self-checking testbench. verify_rtl.py (or gen_coeffs.py --test N) writes coeffs.mem,
// stim.mem, expected.mem and params.vh. Every accepted output is also written to
// rtl_out.txt (one signed decimal per line) so it can be compared outside the simulator. Input valid and output ready are randomly
// throttled so the handshake and stall logic are exercised.
`timescale 1ns/1ps
module tb_polyphase_resampler;
    `include "params.vh"

    logic clk = 0, rst = 1;
    logic signed [DATA_W-1:0] s_data, m_data;
    logic s_valid = 0, s_ready, m_valid, m_ready = 0;

    logic signed [DATA_W-1:0] stim [N_IN];
    logic signed [DATA_W-1:0] expd [N_OUT];
    int n_in = 0, n_out = 0, errors = 0, stall_pct = 30, fd;

    polyphase_resampler #(.L(L), .M(M), .K(K), .DATA_W(DATA_W), .COEF_W(COEF_W), .COEF_FRAC(COEF_FRAC),
                          .COEF_FILE("coeffs.mem")) dut (.*);

    always #5 clk = ~clk;
    assign s_data = (n_in < N_IN) ? stim[n_in] : '0;

    initial begin
        $readmemh("stim.mem", stim);
        $readmemh("expected.mem", expd);
        fd = $fopen("rtl_out.txt", "w");
        if ($value$plusargs("stall=%d", stall_pct)) $display("stall probability %0d%%", stall_pct);
        repeat (4) @(negedge clk);
        rst = 0;
    end

    always @(posedge clk) begin
        if (!rst) begin
            if (s_valid && s_ready) n_in <= n_in + 1;
            if (m_valid && m_ready) begin
                $fdisplay(fd, "%0d", m_data);
                if (m_data !== expd[n_out]) begin
                    errors <= errors + 1;
                    if (errors < 10) $display("MISMATCH y[%0d]: got %0d, expected %0d", n_out, m_data, expd[n_out]);
                end
                n_out <= n_out + 1;
            end
            // new random throttle for the next cycle
            s_valid <= ((n_in + ((s_valid && s_ready) ? 1 : 0)) < N_IN) && ($urandom_range(99) >= stall_pct);
            m_ready <= ($urandom_range(99) >= stall_pct);
        end
    end

    initial begin
        wait (n_out == N_OUT);
        @(posedge clk);
        if (errors == 0) $display("PASS: %0d outputs match (L=%0d, M=%0d, K=%0d, %0d inputs)", N_OUT, L, M, K, n_in);
        else             $display("FAIL: %0d of %0d outputs differ", errors, N_OUT);
        $fclose(fd);
        $finish;
    end

    initial begin
        #(200_000_000);
        $display("FAIL: timeout after %0d of %0d outputs (%0d inputs accepted)", n_out, N_OUT, n_in);
        $fclose(fd);
        $finish;
    end
endmodule
