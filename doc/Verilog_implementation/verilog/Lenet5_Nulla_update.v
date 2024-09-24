//`timescale 1ns/1ps
module nullaLayers(in, out, clk, Start, Reset, Qi, Qcomp, Qd);
	input [1175 : 0] in;
	input clk;
	output [83 : 0] out;

	reg [1175 : 0] reg0_mem_read_data;
	reg o_2_0, o_2_1, o_2_2, o_2_3, o_2_4, o_2_5, o_2_6, o_2_7, o_2_8, o_2_9, o_2_10, o_2_11, o_2_12, o_2_13, o_2_14, o_2_15;
	reg i_2_0, i_2_1, i_2_2, i_2_3, i_2_4, i_2_5, i_2_6, i_2_7, i_2_8, i_2_9, i_2_10, i_2_11, i_2_12, i_2_13, i_2_14, i_2_15, i_2_16, i_2_17, i_2_18, i_2_19, i_2_20, i_2_21, i_2_22, i_2_23, i_2_24, i_2_25, i_2_26, i_2_27, i_2_28, i_2_29, i_2_30, i_2_31, i_2_32, i_2_33, i_2_34, i_2_35, i_2_36, i_2_37, i_2_38, i_2_39, i_2_40, i_2_41, i_2_42, i_2_43, i_2_44, i_2_45, i_2_46, i_2_47, i_2_48, i_2_49, i_2_50,i_2_51, i_2_52, i_2_53, i_2_54, i_2_55, i_2_56, i_2_57, i_2_58, i_2_59, i_2_60, i_2_61, i_2_62, i_2_63, i_2_64, i_2_65, i_2_66, i_2_67, i_2_68, i_2_69, i_2_70, i_2_71, i_2_72, i_2_73, i_2_74, i_2_75, i_2_76, i_2_77, i_2_78, i_2_79, i_2_80, i_2_81, i_2_82, i_2_83, i_2_84, i_2_85, i_2_86, i_2_87, i_2_88, i_2_89, i_2_90, i_2_91, i_2_92, i_2_93, i_2_94, i_2_95, i_2_96, i_2_97, i_2_98, i_2_99, i_2_100, i_2_101, i_2_102, i_2_103, i_2_104, i_2_105, i_2_106, i_2_107, i_2_108, i_2_109, i_2_110, i_2_111, i_2_112, i_2_113, i_2_114, i_2_115, i_2_116, i_2_117, i_2_118, i_2_119, i_2_120, i_2_121, i_2_122, i_2_123, i_2_124, i_2_125, i_2_126, i_2_127, i_2_128, i_2_129, i_2_130, i_2_131, i_2_132, i_2_133, i_2_134, i_2_135, i_2_136, i_2_137, i_2_138, i_2_139, i_2_140, i_2_141, i_2_142, i_2_143, i_2_144, i_2_145, i_2_146, i_2_147, i_2_148, i_2_149;
	//wire [1175 : 0] reg0_mem_write_data;
	//wire [1175 : 0] reg0_write_data;
	reg [399 : 0] reg1_mem_read_data;
	//wire [399 : 0] reg1_mem_write_data;
	wire [1599 : 0] reg1_write_data;
	//wire [399 : 0] reg1_write_flatten_data;
	//reg [119 : 0] reg2_mem_read_data;
	//wire [119 : 0] reg2_mem_write_data;
	//wire [119 : 0] reg2_write_data;
	//reg [83 : 0] reg3_mem_read_data;
	//wire [83 : 0] reg3_mem_write_data;
	//wire [83 : 0] reg3_write_data;
input Start, Reset;
output  Qi, Qcomp, Qd;
reg [2:0] state;
reg [3:0] I;//count for read register
reg [3:0] J;//count for write register
reg [3:0] count;//counter for num of kernels
reg[3:0] row_compl; //Flag to increment every 10 clks

localparam 
INI  = 	5'b00001, // "Initial" state
COMP = 	5'b00010, // "Instantiate" state
DONE = 	5'b00100, // "DONE" state
num_ch = 6,
out_row = 10,
num_rows = 14;

assign {Qd, Qcomp, Qi} = state;
kernel_2 kernel_2_all(i_2_0, i_2_1, i_2_2, i_2_3, i_2_4, i_2_5, i_2_6, i_2_7, i_2_8, i_2_9, i_2_10, i_2_11, i_2_12, i_2_13, i_2_14, i_2_15, i_2_16, i_2_17, i_2_18, i_2_19, i_2_20, i_2_21, i_2_22, i_2_23, i_2_24, i_2_25, i_2_26, i_2_27, i_2_28, i_2_29, i_2_30, i_2_31, i_2_32, i_2_33, i_2_34, i_2_35, i_2_36, i_2_37, i_2_38, i_2_39, i_2_40, i_2_41, i_2_42, i_2_43, i_2_44, i_2_45, i_2_46, i_2_47, i_2_48, i_2_49, i_2_50, i_2_51, i_2_52, i_2_53, i_2_54, i_2_55, i_2_56, i_2_57, i_2_58, i_2_59, i_2_60, i_2_61, i_2_62, i_2_63, i_2_64, i_2_65, i_2_66, i_2_67, i_2_68, i_2_69, i_2_70, i_2_71, i_2_72, i_2_73, i_2_74, i_2_75, i_2_76, i_2_77, i_2_78, i_2_79, i_2_80, i_2_81, i_2_82, i_2_83, i_2_84, i_2_85, i_2_86, i_2_87, i_2_88, i_2_89, i_2_90, i_2_91, i_2_92, i_2_93, i_2_94, i_2_95, i_2_96, i_2_97, i_2_98, i_2_99, i_2_100, i_2_101, i_2_102, i_2_103, i_2_104, i_2_105, i_2_106, i_2_107, i_2_108, i_2_109, i_2_110, i_2_111, i_2_112, i_2_113, i_2_114, i_2_115, i_2_116, i_2_117, i_2_118, i_2_119, i_2_120, i_2_121, i_2_122, i_2_123, i_2_124, i_2_125, i_2_126, i_2_127, i_2_128, i_2_129, i_2_130, i_2_131, i_2_132, i_2_133, i_2_134, i_2_135, i_2_136, i_2_137, i_2_138, i_2_139, i_2_140, i_2_141, i_2_142, i_2_143, i_2_144, i_2_145, i_2_146, i_2_147, i_2_148, i_2_149, o_2_0, o_2_1, o_2_2, o_2_3, o_2_4, o_2_5, o_2_6, o_2_7, o_2_8, o_2_9, o_2_10, o_2_11, o_2_12, o_2_13, o_2_14, o_2_15);




always @(posedge clk, posedge Reset) 

  begin  : kernel2
    if (Reset)
       begin
         state <= INI;
         I <= 4'bXXXX;   // to avoid recirculating mux controlled by Reset 
	     J <= 4'bXXXX;
	     count <= 4'bXXXX;
		 row_compl <= 4'bXXXX;
		 
	    end
	else
       begin
           case (state)
			INI	: 
	          begin
		         // RTL operations in the Data Path            	              
		        I <= 0;
				J <= 0;
				count <= 1; //1 because plus 30 before I becomes 10
				row_compl <= 0;
		         // state transitions in the control unit
		         if (Start)
		           state <= COMP;
	          end
			  
			COMP	:
	          begin
					//i_2_0 <= reg0_mem_read_data[6*I+24*F];//14*6*F, F= one entire row complete
					i_2_0 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*I];//reset I every 10 clocks or 25 clocks
					//num_rows = 14, num_ch = 6, row_compl= count to indicate one row completion, I=read register counter
					i_2_1 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_2 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_3 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_4 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_5 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_6 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_7 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_8 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_9 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_10 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_11 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_12 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_13 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_14 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_15 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_16 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_17 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_18 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_19 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_20 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_21 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+17)+num_ch*out_row*4];
					i_2_22 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+18)+num_ch*out_row*4];
					i_2_23 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+19)+num_ch*out_row*4];
					i_2_24 <= reg0_mem_read_data[num_rows*num_ch*row_compl+num_ch*(I+20)+num_ch*out_row*4];
					
					i_2_25 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*I]; //resets every 25
					i_2_26 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_27 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_28 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_29 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_30 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_31 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_32 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_33 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_34 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_35 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_36 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_37 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_38 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_39 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_40 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_41 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_42 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_43 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_44 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_45 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_46 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+17)+num_ch*out_row*4];
					i_2_47 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+18)+num_ch*out_row*4];
					i_2_48 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+19)+num_ch*out_row*4];
					i_2_49 <= reg0_mem_read_data[1+num_rows*num_ch*row_compl+num_ch*(I+20)+num_ch*out_row*4];
					
					i_2_50 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*I]; //resets every 25
					i_2_51 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_52 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_53 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_54 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_55 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_56 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_57 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_58 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_59 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_60 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_61 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_62 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_63 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_64 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_65 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_66 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_67 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_68 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_69 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_70 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_71 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_72 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_73 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_74 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					
					i_2_75 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*I]; //resets every 25
					i_2_76 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_77 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_78 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_79 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_80 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_81 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_82 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_83 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_84 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_85 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_86 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_87 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_88 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_89 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_90 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_91 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_92 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_93 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_94 <= reg0_mem_read_data[2+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_95 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_96 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+17)+num_ch*out_row*4];
					i_2_97 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+18)+num_ch*out_row*4];
					i_2_98 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+19)+num_ch*out_row*4];
					i_2_99 <= reg0_mem_read_data[3+num_rows*num_ch*row_compl+num_ch*(I+20)+num_ch*out_row*4];
					
					i_2_100 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*I]; //resets every 25
					i_2_101 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_102 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_103 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_104 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_105 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_106 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_107 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_108 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_109 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_110 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_111 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_112 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_113 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_114 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_115 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_116 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_117 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_118 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_119 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_120 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_121 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+17)+num_ch*out_row*4];
					i_2_122 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+18)+num_ch*out_row*4];
					i_2_123 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+19)+num_ch*out_row*4];
					i_2_124 <= reg0_mem_read_data[4+num_rows*num_ch*row_compl+num_ch*(I+20)+num_ch*out_row*4];
					
					i_2_125 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*I]; //resets every 25
					i_2_126 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+1)];
					i_2_127 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+2)];
					i_2_128 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+3)];
					i_2_129 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+4)];
					
					i_2_130 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+4)+num_ch*out_row];
					i_2_131 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+5)+num_ch*out_row];
					i_2_132 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+6)+num_ch*out_row];
					i_2_133 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+7)+num_ch*out_row];
					i_2_134 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row];
					
					i_2_135 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+8)+num_ch*out_row*2];
					i_2_136 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+9)+num_ch*out_row*2];
					i_2_137 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+10)+num_ch*out_row*2];
					i_2_138 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+11)+num_ch*out_row*2];
					i_2_139 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*2];
					
					i_2_140 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+12)+num_ch*out_row*3];
					i_2_141 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+13)+num_ch*out_row*3];
					i_2_142 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+14)+num_ch*out_row*3];
					i_2_143 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+15)+num_ch*out_row*3];
					i_2_144 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*3];
					
					i_2_145 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+16)+num_ch*out_row*4];
					i_2_146 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+17)+num_ch*out_row*4];
					i_2_147 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+18)+num_ch*out_row*4];
					i_2_148 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+19)+num_ch*out_row*4];
					i_2_149 <= reg0_mem_read_data[5+num_rows*num_ch*row_compl+num_ch*(I+20)+num_ch*out_row*4];
					
					o_2_0 <= reg1_write_data[J];
					o_2_1 <= reg1_write_data[J+1]; 
					o_2_2 <= reg1_write_data[J+2];  
					o_2_3 <= reg1_write_data[J+3]; 
					o_2_4 <= reg1_write_data[J+4]; 
					o_2_5 <= reg1_write_data[J+5]; 
					o_2_6 <= reg1_write_data[J+6]; 
					o_2_7 <= reg1_write_data[J+7]; 
					o_2_8 <= reg1_write_data[J+8]; 
					o_2_9 <= reg1_write_data[J+9]; 
					o_2_10 <= reg1_write_data[J+10]; 
					o_2_11 <= reg1_write_data[J+11];  
					o_2_12 <= reg1_write_data[J+12]; 
					o_2_13 <= reg1_write_data[J+13]; 
					o_2_14 <= reg1_write_data[J+14]; 
					o_2_15 <= reg1_write_data[J+15]; 
					
					if(count != 100) begin
						I <= I + 1;
						J <= J + 16;//16 = output channels
						count <= count + 1;
					end
					
					if(count % 10 == 0)
						row_compl <= row_compl + 1; //counter when one row completes, move to next row
						
					if((count % 10 == 0) || (count % 25 == 0))
						I <= 0;
						
		            if(count == 100)
						state <= DONE;
 	          end
			  
			DONE	:
	          begin  
		         
		           state <= INI; // Transit to INI state unconditionally
		       end    
		   endcase
		end
    end 
endmodule
