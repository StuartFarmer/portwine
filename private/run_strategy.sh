#!/bin/bash

# Log file for the strategy
LOG_FILE="strategy.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run the strategy
run_strategy() {
    log "Starting strategy..."
    poetry run python gap_down_strategy.py
    EXIT_CODE=$?
    log "Strategy exited with code $EXIT_CODE"
    return $EXIT_CODE
}

# Create log file if it doesn't exist
touch "$LOG_FILE"

# Main loop
while true; do
    run_strategy
    
    # If the strategy exited with a non-zero code, wait before restarting
    if [ $? -ne 0 ]; then
        log "Strategy crashed. Waiting 60 seconds before restarting..."
        sleep 60
    else
        log "Strategy completed successfully. Restarting..."
    fi
done 