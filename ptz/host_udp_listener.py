import socket
import json

HOST = "0.0.0.0"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening on {HOST}:{PORT}")

while True:
    data, addr = sock.recvfrom(4096)
    msg = json.loads(data.decode("utf-8"))
    print(f"From {addr}: {msg}")
