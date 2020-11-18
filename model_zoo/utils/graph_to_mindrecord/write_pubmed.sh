#!/bin/bash
SRC_PATH=/tmp/pubmed/dataset
MINDRECORD_PATH=/tmp/pubmed/mindrecord

rm -f $MINDRECORD_PATH/*

python writer.py --mindrecord_script pubmed \
--mindrecord_file "$MINDRECORD_PATH/pubmed_mr" \
--mindrecord_partitions 1 \
--mindrecord_header_size_by_bit 18 \
--mindrecord_page_size_by_bit 20 \
--graph_api_args "$SRC_PATH"
