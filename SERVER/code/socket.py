import socket, sys, os.path, errno, json, io, pickle

HEADERSIZE = 10
# server_socket = None

def connect_to(host, port, server = None):
    server_socket = None
    if not server:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("socket created!")
        server_socket.bind((host, port))
        print(f"Listening to {host}:{port}")
        server_socket.listen(1)
    else:
        server_socket = server
        print("socket re-assigned")
    client_socket = None

    # try:
    # initialize socket
    
    try:
        client_socket, address = server_socket.accept()
        print(server_socket)
        print(f"Connected to {address[0]}:{address[1]}")
    except OSError as we_aint_good:
        print(we_aint_good)
    
    # get all the data sent by client: dict {csv file, json file}
    
    return client_socket, server_socket

def read_header(data_stream):
    try:
        return int(data_stream[:HEADERSIZE])
    except Exception as e:
        print(e) 

def receive_on_ram(client_socket):
    data=b''
    new_msg=True
    msglen=sys.maxsize
    # receives data by reading length header
    while len(data)-HEADERSIZE < msglen:
        chunk = client_socket.recv(1024)

        if new_msg:
            # print("new msg len:",chunk[:HEADERSIZE])
            msglen = read_header(chunk)
            if msglen <= 0:
                return (-1, None), (-1, None)
            new_msg = False

        data += chunk
    # unpack serialized json dict
    try:
        received = (pickle.loads(data[HEADERSIZE:]))
    except Exception as e:
        return (-1, None), (-1, None)
    # access each file content
    rc1 = received['data']
    rj = received['json']
    return (1, rc1), (1, rj)
    # socket close commands to be executed in calling function
    # finally:
    #     if client_socket:
    #         client_socket.close()
    #         server_socket.close()

if __name__ == "__main__":
    print("Dont call this file as main")
