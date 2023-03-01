#!/usr/bin/env python3
import socket
from _thread import start_new_thread
import threading


HOST = "127.0.0.1"
PORT = 65432


def threaded(conn, lock):
    continue_loop = True
    while continue_loop:
        data = conn.recv(1024)
        if not data:
            print("Bye")
            lock.release()  # release lock before breaking
            continue_loop = False
        conn.send(data)  # send data back to client
    conn.close()


def main():
    print_lock = threading.Lock()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("Socket binded to port")
        s.listen()
        while True:
            conn, addr = s.accept()
            print_lock.acquire()
            print("Connected to:", addr[0], ":", addr[1])
            start_new_thread(threaded, (conn, print_lock,))


if __name__ == "__main__":
    main()
