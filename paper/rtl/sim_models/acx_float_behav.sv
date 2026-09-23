// Simulation stand-in for ACX_FLOAT (an unconnected pin).  The vendor sim model
// drives 1'bz (speedster7t_sim_io.sv:665-670); Verilator is two-state, so this
// drives 0.  Every cascade or BRAM input the chain leaves "Open" is either unused
// by the behavioural models or ignored by the selected configuration.
`ifndef SYNTHESIS
module ACX_FLOAT (y);
    output y;
    assign y = 1'b0;
endmodule
`endif
