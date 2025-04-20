#!/bin/bash

# Mind the trailing slash! We're copying the _contents_ inside $src into $dst.
src='/compute/babel-4-1/yusenh/'
dst='/scratch/yusenh'

# Record the current node name, so we can 
node=$(hostname -s)
nodes_list="$HOME/nodes_with_our_scratch_data"
if ! grep -q $node $nodes_list; then
    echo $node >> $nodes_list
fi

# TODO: if two SLURM jobs are scheduled to the same node,
# they will both try to copy the data at the same time.
# We need to use a lock file to prevent this.

# TODO: determine if data has already been copied onto
# the current node. If so, skip the copy.

echo "Copying data in $src to $dst"
mkdir -p $dst
rsync -ah --info=progress2 $src $dst
