#!/usr/bin/env python3
import socket
from _thread import start_new_thread
import threading
import pickle
import analysis


HOST = "127.0.0.1"
PORT = 65432

def get_price(stock_name, day):
    price = analysis.get_price(stock_name, day)
    return price

def get_daily_returns(stock_name, days):
    returns = analysis.get_daily_returns(stock_name, days)
    return returns

def get_avg_daily_returns(stock_name, days):
    returns = analysis.get_avg_daily_return(stock_name, days)
    return returns

def get_std(stock_name, days):
    std = analysis.get_std(stock_name, days)
    return std

def get_sharpe(stock_name, days, rfr):
    sharpe = analysis.get_sharpe(stock_name, days, rfr)
    return sharpe

def get_portfolio_returns(stocks, days):
    returns = analysis.get_portfolio_returns(stocks, days)
    return returns

def min_var_portfolio(stocks, days):
    combined_returns = analysis.get_portfolio_returns(stocks, days)
    opt_weights, min_var = analysis.min_var_portfolio(combined_returns)
    
    data_out = opt_weights.tolist()
    data_out.append(min_var)

    return data_out

def max_sharpe_portfolio(stocks, days, rfr):
    combined_returns = analysis.get_portfolio_returns(stocks, days)
    opt_weights, max_sharpe = analysis.max_sharpe_portfolio(combined_returns, rfr)

    data_out = opt_weights.tolist()
    data_out.append(max_sharpe)

    return data_out

def client_connection_thread(conn, port, lock):
    continue_loop = True
    while continue_loop:
        data = conn.recv(1024)
        if data:
            decoded = pickle.loads(data)
            print("Received:", decoded)
            choice = str(decoded.pop())
            days = int(decoded.pop())
            stock_name = decoded[0]
            if choice == '1': #Predict price of stock
                data_out = get_price(stock_name, days)

            elif choice == '2': #Predict returns of (single) stock
                data_out = get_daily_returns(stock_name, days)

            elif choice == '3':
                data_out = get_avg_daily_returns(stock_name, days)

            elif choice == '4':
                data_out = get_std(stock_name, days)

            elif choice == '5':
                rfr = float(decoded.pop())
                data_out = get_sharpe(stock_name, days, rfr)

            elif choice == '6':
                data_out = get_portfolio_returns(decoded, days)

            elif choice == '7':
                data_out = min_var_portfolio(decoded, days)
            
            elif choice == '8':
                rfr = float(decoded.pop())
                data_out = max_sharpe_portfolio(decoded, days, rfr)
        else:
            print(f"Closing connection on port {port}")
            lock.release()  # release lock before breaking
            continue_loop = False

        data_pickle = pickle.dumps(data_out)
        conn.send(data_pickle)  # send data back to client
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
