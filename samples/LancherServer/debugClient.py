# -*- coding:utf-8 -*-
import socket

def main(addr, port):
    try:
        while True:
            print(">",end="")
            cmd = input();

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(1.0);
            client.connect((addr, port))

            client.send(cmd.encode()) 
            client.send("".encode())
            try:
                response = client.recv(4096) 
                print(response)
            except socket.timeout:
                print("no response")
            client.close()

        except KeyboardInterrupt:
            print("citl-c interrupt")

if __name__ == '__main__':
    main("192.168.100.57",8000)
