#!/bin/bash
pip install -r "requirements.txt"
echo "Starting server"
python3 server.py> logs/server_tf.log 2> logs/server.log &
sleep 10 # Sleep for 3s to give the server enough time to start
for letter in {{A..Y},ZA,ZB,ZC}
do
    echo "Starting client $letter"
    python3 client.py --client $letter> logs/${letter}_tf.log 2> logs/${letter}.log &
done
# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
