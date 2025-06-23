#!/bin/bash

OUTPUT_FILE="memory_usage.csv"

# Create CSV header if it doesn't exist
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "timestamp,total(GB),used(GB),free(GB)" >> "$OUTPUT_FILE"
fi

echo "Starting memory usage recording every 5 seconds..."
echo "Log file: $OUTPUT_FILE"

while true; do
    CURRENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")

    # Use 'free -h' to get actual memory (with the 'available' column)
    read -r TOTAL USED FREE <<< $(free -h --si | awk '/^Mem:/ {print $2, $3, $7}')

    # Remove non-numeric characters
    TOTAL_NUM=$(echo "$TOTAL" | sed 's/[^0-9.]//g')
    USED_NUM=$(echo "$USED" | sed 's/[^0-9.]//g')
    FREE_NUM=$(echo "$FREE" | sed 's/[^0-9.]//g')

    echo "$CURRENT_DATE,$TOTAL_NUM,$USED_NUM,$FREE_NUM" >> "$OUTPUT_FILE"

    sleep 5
done
