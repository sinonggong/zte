// ------------------------------------------------------------------
//
// Copyright (c) 2021  Achronix Semiconductor Corp.
// All Rights Reserved.
//
//
// This software constitutes an unpublished work and contains
// valuable proprietary information and trade secrets belonging
// to Achronix Semiconductor Corp.
//
// This software may not be used, copied, distributed or disclosed
// without specific prior written authorization from
// Achronix Semiconductor Corp.
//
// The copyright notice above does not evidence any actual or intended
// publication of such software.
//
// ------------------------------------------------------------------
// Master reset processor V2
//      Combines an array of negative sense reset signals, 
//      The input stage can optionally include a synchronizer and pipeline
//      per reset if SYNC_INPUT_RESETS is enabled
//      If not, then set_multicycle path will be required on each input
//      to allow for propogation from the edge of the fabric to the centre
//      For the output there is again either a pipeline to allow for 
//      distance and fanout across the die, or else a single synchronizer
//      if RESET_OVER_CLOCK is enabled
// ------------------------------------------------------------------


module reset_processor_v2
#(
    parameter   NUM_INPUT_RESETS        = 0,        // Number of clocks, (and hence resets).
    parameter   IN_RST_PIPE_LENGTH      = 2,        // Length of input flop pipeline, minimum of 2
                                                    // Ignored if SYNC_INPUT_RESETS = 0
    parameter   SYNC_INPUT_RESETS       = 0,
    parameter   OUT_RST_PIPE_LENGTH     = 2,        // Length of reset flop pipeline, minimum of 2
                                                    // Ignored if RESET_OVER_CLOCK = 1
    parameter   RESET_OVER_CLOCK        = 0         // Will the output reset be routed over the clock network
                                                    // If so, then the output signal needs to have a
                                                    // set_clock_type -data_center [get_nets <output_net>]
                                                    // set in the project .pdc file
)
(
    // Inputs
    input  wire [NUM_INPUT_RESETS  -1:0] i_rstn_array,  // Array of negative sense reset inputs
    input  wire                          i_clk,         // Input clock array

    // Outputs
    output wire                          o_rstn         // Processed resets
);


    // Combine the input array into one master reset
    logic master_rstn;

    generate if (SYNC_INPUT_RESETS) begin : gb_sync_inputs
        logic [NUM_INPUT_RESETS -1:0] sync_rstn_pipe_out;

        for (genvar ii=0; ii < NUM_INPUT_RESETS; ii = ii + 1) begin : gb_per_input
            logic                           sync_rstn;
            logic [IN_RST_PIPE_LENGTH -1:0] sync_rstn_pipe;

            ACX_SYNCHRONIZER x_sync (.din(1'b1), .dout(sync_rstn), .clk(i_clk), .rstn(i_rstn_array[ii]));

            always @(posedge i_clk)
                sync_rstn_pipe <= {sync_rstn_pipe[IN_RST_PIPE_LENGTH-2:0], sync_rstn};

            assign sync_rstn_pipe_out[ii] = sync_rstn_pipe[IN_RST_PIPE_LENGTH-1];
        end
        // Code to not propogate X
        assign master_rstn = (sync_rstn_pipe_out === {NUM_INPUT_RESETS{1'b1}}) ? 1'b1 : 1'b0;
    end
    else 
    begin : gb_async_inputs
        assign master_rstn = (i_rstn_array === {NUM_INPUT_RESETS{1'b1}}) ? 1'b1 : 1'b0;
    end
    endgenerate

    generate if (RESET_OVER_CLOCK) begin : gb_roc
        // If reset over clock required, use a further synchronizer to ensure there is the single retained flop
        // that is not retimed or duplicated
        ACX_SYNCHRONIZER x_sync_rstn  (.din(1'b1), .dout(o_rstn),  .clk(i_clk), .rstn(master_rstn));
    end
    else
    begin : gb_out_pipe
        // Create an output pipeline
        // Set a default maxfan to have Synplify duplicate the necessary flops
        logic [OUT_RST_PIPE_LENGTH -1:0] sync_rstn_out_pipe  = {OUT_RST_PIPE_LENGTH{1'b0}} /* synthesis syn_maxfan=16 */;

        always @(posedge i_clk)
            sync_rstn_out_pipe <= {sync_rstn_out_pipe[OUT_RST_PIPE_LENGTH-2:0], master_rstn};

        assign o_rstn = sync_rstn_out_pipe[OUT_RST_PIPE_LENGTH-1];
    end
    endgenerate

endmodule : reset_processor_v2

