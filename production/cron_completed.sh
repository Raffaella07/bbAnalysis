#!/bin/bash

# Check if JOBARRAY_ID is provided as input
if [ -z "$1" ]; then
    echo "Please provide the JOBARRAY_ID as input."
    exit 1
fi

JOBARRAY_ID="$1"

# Get the list of completed ARRAY IDs
completed_array_ids=$(sacct -j "$JOBARRAY_ID" --format=JobID%30,State | grep "COMPLETED" | awk '{print $1}' | cut -d_ -f2 | cut -d. -f1)
#echo $completed_array_ids
# Delete the log files for completed jobs
for array_id in $completed_array_ids; do
    err_file="/t3home/ratramon/EXO-MCsampleProductions/FullSimulation/RunIISummer20UL18/production/err/cms_sim_"$JOBARRAY_ID"_"$array_id".err"
    out_file="/t3home/ratramon/EXO-MCsampleProductions/FullSimulation/RunIISummer20UL18/production/out/cms_sim_"$JOBARRAY_ID"_"$array_id".out"
     
    if [ -f "$err_file" ]; then
         rm "$err_file"
    fi

    if [ -f "$out_file" ]; then
         rm "$out_file"
    fi
done