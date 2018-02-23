# -*- coding:utf-8 -*-
import socket
import sysv_ipc
import subprocess
from subprocess import Popen
from time import sleep


sharedMemoryKey = 314159

def isRunning():
    cmd = "ps aux | grep VrStreamer | grep -v grep | wc -l"
    res = subprocess.Popen( cmd, shell  = True,  stdin  = subprocess.PIPE,  stdout = subprocess.PIPE, stderr = subprocess.PIPE) 
    stdout_data, stderr_data = res.communicate() #処理実行を待つ(†1)
    result = stdout_data.decode()
    print( stdout_data)
    if(b'0\n' != result):
        return True
    else:
        return False
    
def main(addr, port):
    #phase 0 initial state
    #phase 1 setup ipaddress
    clientIpAddress = None
    phase = 0
    status = "free"
    #memory initialize
    memory = sysv_ipc.SharedMemory(sharedMemoryKey, sysv_ipc.IPC_CREAT,size = 4 )
   # memory_value = memory.read()
    #establish server
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((addr,port)) #IPとPORTを指定してバインドします
    serversock.listen(10) #接続の待ち受けをします（キューの最大数を指定）

    proc = None;
    try:
        while True:
            print('>', end="")
            clientsock, client_address = serversock.accept() #接続されればデータを格納

            rawdata = clientsock.recv(32)
            recievedMessage = rawdata.decode();
            #        print( 'Received -> %s' % rcvmsg)
            tokens = recievedMessage.split(" ");
            print("> "+recievedMessage)
            
            if(tokens[0] == "hello"):
                #None
                print("hello");

            #request current server status
            elif(tokens[0] == "status"):
                if(status == "free"):
                    clientsock.send("free".encode())
                    print("status is free")
                elif(status == "runnning"):
                    clientsock.send("running".encode())
                    print("status is running \n server is already runnning")

            #request current client ip
            elif(tokens[0] == "clientIp"):
                #register ip address
                if(clientIpAddress == None):
                    clientsock.send("Null".encode())
                    print("client ip not yet registerd")
                else:
                    clientsock.send(clientIpAddress.encode())
                    print("current client ip is " + clientIpAddress)

            #resist client ip address
            elif(tokens[0] == "registerIp"):
                if(tokens[1] == None):
                    clientsock.send("no".encode())
                    print("clientIpaddress is None" + clientIpAddress)

                clientIpAddress = tokens[1]
                clientsock.send("ok".encode())
                print("set clientIpaddress" + clientIpAddress)

            #change server status
            elif(tokens[0] == "run" ):
                
                if(not isRunning()):
                    print("server is already runnning")
                    clientsock.send("fail".encode())
                elif(clientIpAddress == None):
                    print("client ip not yet registerd")
                    clientsock.send("fail".encode())
                else:
                    try:
                        memory.write(chr(1))
                    except:
                        print("error")
                    print("start Streaming server...")
                    cmd = "../VrStreamer "+"-camid 0 -share " +str(sharedMemoryKey)
                    proc = Popen( cmd,shell=True )
                    clientsock.send("succeed".encode())
                
                
            elif(tokens[0] == "stop" ):
                memory.write(chr(0))
                sleep(1)
                if(isRunning == False):
                    clientsock.send("succeed".encode())
                else:
                    clientsock.send("fail".encode())

                print("stop Streaming server...")

            elif(tokens[0] == "restart" ):
                memory.write(chr(1))
                print("restarting Streaming server...")                
                
            elif(tokens[0] == "exit"):
                #kill gstreamer process 
                # memory.write(0)
                print("stop Streaming server...")

                print("exit")
                break
                
    except KeyboardInterrupt:
        print("citl-c interrupt")

    memory.detach()        
    serversock.close()

if __name__ == '__main__':
    main("192.168.100.57",8000)
