import io
import numpy as np
import socket, sys, os, errno, pickle
from code.socket import * 
from code.machawai_onnx import onnx_inf

HEADERSIZE = 10

host='0.0.0.0'
port=8888
# Over 0.0.0.0: May be changed

def pipe_to_model():
    # Initialize TCP socket to start communication

    print("\nThe model needs both a\ndata CSV file\nand a\nfeatures JSON file\nPlease provide it")
    
    return connect_to(host, port) 

def model_interaction(c):
    # module for single interaction:
    # can implement iterations over calling this
    try:
        data_input, json_input= receive_on_ram(c)
    except:
        data_input, json_input = [-1], [-1]
    if data_input[0] <= 0 and json_input[0] <= 0:
        return -1
    results = onnx_inf((data_input[1], json_input[1]))
    # serializing results so that client can draw graph
    sample = pickle.dumps(results[0])
    sample = bytes(f"{len(sample):<{HEADERSIZE}}", "utf-8")+sample
    # sample = sample+bytes(f"{'*'*(1024-len(sample)%1024)}","utf-8")

    pred = pickle.dumps(results[1])
    pred = bytes(f"{len(pred):<{HEADERSIZE}}", "utf-8")+pred
    
    # sends results over TCP channel
    print("\nProcessed results on their way back to the client!\n")
    try:
        c.sendall(sample)
        c.sendall(pred)
    except:
        # client.close()
        # server.close()
        print("Error!") # put proper error handling
        return -1
    return 0

if __name__ == "__main__":
    client, server = pipe_to_model()
    while 1:
        while model_interaction(client) == 0:
            continue
        client.close()
        client, server = connect_to(host, port, server)
        # server.close()
