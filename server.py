#!/usr/bin/env python3

import socket

from _thread import *
import threading

print_lock = threading.Lock()


#Thread function
def threaded(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            print('Bye')
            #release lock before breaking
            print_lock.release()
            break

        #Send data back to client
        conn.send(data)
    conn.close()

def Main():
    HOST = '127.0.0.1'
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("Socket binded to port")

        s.listen()

        while True:
            conn, addr = s.accept()
            print_lock.acquire()
            print("Connected to:", addr[0], ':', addr[1])

            start_new_thread(threaded, (conn,))
        

if __name__ == '__main__':
    Main()