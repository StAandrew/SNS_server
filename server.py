#!/usr/bin/env python3
import socket
from _thread import start_new_thread
import threading


HOST = "127.0.0.1"
PORT = 65433

def fetch_stock(stock_name):
    #FETCH STOCK CODE HERE
    stock_data = "Data about %s." % (stock_name)
    return stock_data

def predict_stock(stock_name):
    #PREDICT STOCK CODE HERE
    stock_predictions = "Predictions about %s" % (stock_name)
    return stock_predictions

def daily_rundown():
    #DAILY RUNDOWN CODE HERE
    rundown = "Daily rundown here"
    return rundown

def client_connection_thread(conn, port, lock):
    continue_loop = True
    while continue_loop:
        data = conn.recv(1024)
        if data:
            decoded = data.decode().split(',')
            print("Received:", decoded)
            choice, stock_name = decoded
            if choice == '1':
                data_out = fetch_stock(stock_name)
            elif choice == '2':
                data_out = predict_stock(stock_name)
            elif choice == '3':
                data_out = daily_rundown()
        else:
            print(f"Closing connection on port {port}")
            lock.release()  # release lock before breaking
            continue_loop = False
        conn.send(data_out.encode())  # send data back to client
    conn.close()


def main():
    print_lock = threading.Lock()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("Socket binded to port")
        s.listen()
        continue_loop = True
        running_threads = []
        while continue_loop:
            try:
                conn, addr = s.accept()
                print_lock.acquire()
                print("Connected to:", addr[0], ":", addr[1])
                new_thread = threading.Thread(target=client_connection_thread, args=(conn, addr[1], print_lock,), daemon=True)
                new_thread.start()
                running_threads.append(new_thread)
            except KeyboardInterrupt:
                print("Quitting...")
                continue_loop = False
                for thread in running_threads:
                    thread.join()

if __name__ == "__main__":
    main()
