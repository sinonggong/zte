# ace -batch -script_file dump_placement.tcl -script_args <acxprj> <impl> <placed.acxdb> <inst-glob> <out>
# Writes "i:<inst> <site>" per instance matching <inst-glob> (e.g. {u_array.g_node_*}) from a placed/routed database.
set argv_list $::argv
lassign $argv_list prj impl db glob out
restore_project $prj -activeimpl $impl -no_db
restore_impl $db -impl $impl
set fh [open $out w]
foreach inst [find -insts $glob] {
    if {[catch {get_placement $inst} site]} { continue }
    puts $fh "$inst $site"
}
close $fh
puts "PLACE_DUMP_DONE"
