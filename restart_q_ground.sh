#!/bin/sh

signal=KILL

sleep_a_while () {
    sleep 30m
}

while true; do
    # Note: command launched in background:
    /home/pawel/Downloads/QGroundControl.AppImage &

    # Save PID of command just launched:
    last_pid=$!

    # Sleep for a while:
    sleep_a_while
    


    # See if the command is still running, and kill it and sleep more if it is:
    echo $last_pid
    echo 'killing qgc'
    killall -q QGroundControl
    sleep 2s
    #
    
    # Go back to the beginning and launch the command again
done

