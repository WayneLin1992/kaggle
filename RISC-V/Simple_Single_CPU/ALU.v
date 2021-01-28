//*****************************************************************************
// Module Declaration
module ALU
(
    srcA_i,
    srcB_i,
    ALUctrl_i,
    ALUresult_o,
    NZCV_o
);
//*****************************************************************************
    // I/O Port Declaration
    input  [32-1:0] srcA_i;      // 32 bits source A
    input  [32-1:0] srcB_i;      // 32 bits source B
    input  [ 4-1:0] ALUctrl_i;   // 4 bits ALU control signals
    output [32-1:0] ALUresult_o; // 32 bits ALU operation result
    output [ 4-1:0] NZCV_o;      // N: Negative, Z: Zero, C: Carry, V: Overflow

    // Global variables Declaration
    // System
    reg    [33-1:0] result; // ALU operation result (include carry out bit)

    // System conection
    // Output
    assign ALUresult_o = result[31:0];
    assign NZCV_o[3]   = ;       // Sign bit
    assign NZCV_o[2]   = ; // NOR all bits
    assign NZCV_o[1]   = ;       // Carry out bit (only for addition/subtract)
    assign NZCV_o[0]   = ; // Positive/Negative overflow
//*****************************************************************************
// Block : ALU operation with behavioral description
    always @(*)
    begin
        case(ALUctrl_i) 
            4'b0000: result = ;                             // AND
            4'b0001: result = ;                             // OR 
            4'b0010: result = ;                     // Addition (signed)
            4'b0110: result = ;           // Subtract (signed)
            4'b0111: result = ; // Set less than (signed)
            4'b1100: result = ;                          // NOR
            4'b1101: result = ;                          // NAND
            default: result = ;                                               // Default: 0
        endcase
    end
//*****************************************************************************
endmodule