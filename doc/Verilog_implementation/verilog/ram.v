`timescale 1ns/1ps
module ram
#(
  parameter integer DATA_WIDTH    = 10,
  parameter integer ADDR_WIDTH    = 12,
)
(
  input  wire                         clk,
  input  wire                         reset,

  input  wire [ ADDR_WIDTH  -1 : 0 ]  s_read_addr,
  output wire [ DATA_WIDTH  -1 : 0 ]  s_read_data,

  input  wire                         s_write_req,
  input  wire [ ADDR_WIDTH  -1 : 0 ]  s_write_addr,
  input  wire [ DATA_WIDTH  -1 : 0 ]  s_write_data
);

  (* ram_style = "block" *) reg  [ DATA_WIDTH -1 : 0 ] mem [ 0 : 1<<ADDR_WIDTH ];

  always @(posedge clk)
  begin: RAM_WRITE
    if (s_write_req)
      mem[s_write_addr] <= s_write_data;
  end
  assign s_read_data = mem[s_read_addr];
endmodule
