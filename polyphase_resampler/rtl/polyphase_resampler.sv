// -----------------------------------------------------------------------------
// Rational polyphase resampler, rate change L/M.
//
//   y[m] = sum_{j=0}^{K-1} c[p][j] * x[n - j]
//   p = (m*M) mod L          coefficient set (leg) used for output m
//   n = floor(m*M / L)       newest input sample used for output m
//   c[p][j] = L * h[p + j*L] leg p of the prototype h (interpolation gain folded in)
//
// This is the "L legs, output commutator" structure: one shared K-sample delay
// line, and the leg index selects which row of the coefficient ROM is applied.
// No zero-stuffed samples are ever stored or multiplied.
//
// Per output the controller advances the phase by M. Every time the phase wraps
// past L one more input sample is shifted into the delay line, so each output
// needs 0, 1 or several new inputs. The last required input can arrive in the
// same clock that launches the output, so the clock only has to be as fast as
// the higher of the input and output sample rates.
//
// Datapath: K parallel multipliers -> adder -> round/saturate, 4 clocks latency.
// Handshake: valid/ready on both sides; the whole pipeline stalls on m_ready low.
//
// Coefficient file (COEF_FILE, $readmemh): L*K lines of COEF_W-bit two's
// complement hex, leg-major: line p*K + j holds round(L*h[p + j*L] * 2^COEF_FRAC).
// gen_coeffs.py writes it.
//
// For complex (I/Q) data use two instances sharing the same coefficient file.
// -----------------------------------------------------------------------------
module polyphase_resampler #(
    parameter int L         = 3,              // interpolation factor (number of legs)
    parameter int M         = 2,              // decimation factor
    parameter int K         = 16,             // taps per leg = ceil(N_taps / L)
    parameter int DATA_W    = 16,             // sample width, signed
    parameter int COEF_W    = 18,             // coefficient width, signed
    parameter int COEF_FRAC = 16,             // fractional bits in the coefficients
    parameter     COEF_FILE = "coeffs.mem"
) (
    input  logic                     clk,
    input  logic                     rst,      // synchronous, active high

    input  logic signed [DATA_W-1:0] s_data,   // input samples, rate f_in
    input  logic                     s_valid,
    output logic                     s_ready,

    output logic signed [DATA_W-1:0] m_data,   // output samples, rate f_in * L / M
    output logic                     m_valid,
    input  logic                     m_ready
);

    localparam int M_DIV  = M / L;                         // whole inputs consumed per output
    localparam int M_MOD  = M % L;                         // phase step per output
    localparam int PW     = (L > 1) ? $clog2(L) : 1;       // phase width
    localparam int NW     = $clog2(M_DIV + 2);             // counts 0 .. M_DIV+1
    localparam int PROD_W = DATA_W + COEF_W;
    localparam int ACC_W  = PROD_W + $clog2(K) + 1;

    // ------------------------------------------------------------ coefficients
    logic signed [COEF_W-1:0] rom [L*K];
    initial $readmemh(COEF_FILE, rom);

    // ------------------------------------------------------------ control
    logic [PW-1:0] phase;      // leg for the next output
    logic [NW-1:0] need;       // inputs still to shift in before the next output
    logic [PW:0]   phase_sum;
    logic          ce, shift, fire;

    assign ce        = !m_valid || m_ready;                // pipeline may advance
    assign s_ready   = ce && (need != 0);
    assign shift     = s_ready && s_valid;
    assign fire      = ce && ((need == 0) || ((need == 1) && s_valid));
    assign phase_sum = (PW+1)'(phase) + (PW+1)'(M_MOD);

    always_ff @(posedge clk) begin
        if (rst) begin
            phase <= '0;
            need  <= NW'(1);                               // y[0] needs x[0]
        end else if (fire) begin
            if (phase_sum >= (PW+1)'(L)) begin             // phase wrapped: one extra input
                phase <= PW'(phase_sum - (PW+1)'(L));
                need  <= NW'(M_DIV + 1);
            end else begin
                phase <= PW'(phase_sum);
                need  <= NW'(M_DIV);
            end
        end else if (shift) begin
            need <= need - NW'(1);
        end
    end

    // ------------------------------------------------------------ delay line
    // dl[j] holds x[n - j]. Shared by all legs.
    logic signed [DATA_W-1:0] dl [K];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int j = 0; j < K; j++) dl[j] <= '0;
        end else if (shift) begin
            dl[0] <= s_data;
            for (int j = 1; j < K; j++) dl[j] <= dl[j-1];
        end
    end

    // ------------------------------------------------------------ datapath
    logic [PW-1:0]            phase_q;                     // leg of the output in flight
    logic                     v1, v2, v3;
    logic signed [PROD_W-1:0] prod [K];
    logic signed [ACC_W-1:0]  sum_c, acc, rounded;
    logic signed [ACC_W-1:0]  shifted;

    always_comb begin
        sum_c = '0;
        for (int j = 0; j < K; j++) sum_c = sum_c + ACC_W'(prod[j]);
    end

    localparam logic signed [ACC_W-1:0] HALF    = (COEF_FRAC > 0) ? (ACC_W'(1) <<< (COEF_FRAC - 1)) : '0;
    localparam logic signed [ACC_W-1:0] MAX_OUT = (ACC_W'(1) <<< (DATA_W - 1)) - ACC_W'(1);
    localparam logic signed [ACC_W-1:0] MIN_OUT = -(ACC_W'(1) <<< (DATA_W - 1));

    assign rounded = acc + HALF;
    assign shifted = rounded >>> COEF_FRAC;                // round half up

    always_ff @(posedge clk) begin
        if (rst) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; m_valid <= 1'b0;
            phase_q <= '0;
        end else if (ce) begin
            // stage 0: remember which leg this output uses (delay line updates on the same edge)
            v1      <= fire;
            phase_q <= phase;
            // stage 1: K multipliers, coefficients of leg phase_q
            v2 <= v1;
            for (int j = 0; j < K; j++)
                prod[j] <= dl[j] * rom[int'(phase_q) * K + j];
            // stage 2: sum of products
            v3  <= v2;
            acc <= sum_c;
            // stage 3: round, saturate
            m_valid <= v3;
            if (shifted > MAX_OUT)      m_data <= MAX_OUT[DATA_W-1:0];
            else if (shifted < MIN_OUT) m_data <= MIN_OUT[DATA_W-1:0];
            else                        m_data <= shifted[DATA_W-1:0];
        end
    end

endmodule
