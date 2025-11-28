import socket
import ssl

HOST = '0.0.0.0'
PORT = 4443
CERTFILE = '/home/vboxuser/project_tls/server.crt'
KEYFILE = '/home/vboxuser/project_tls/server.key'

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)

# Заборона старих протоколів
context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1

with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as bindsock:
    bindsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bindsock.bind((HOST, PORT))
    bindsock.listen(5)
    print(f"[+] Listening on {HOST}:{PORT}")
    while True:
        newsock, addr = bindsock.accept()
        try:
            ssock = context.wrap_socket(newsock, server_side=True)
            print("[+] Connection from", addr)
            print("    TLS version:", ssock.version())
            print("    Cipher:", ssock.cipher())
            # read a small message then respond
            data = ssock.recv(4096)
            if data:
                print("    Received (truncated):", data[:200])
                ssock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello TLS\n")
            ssock.shutdown(socket.SHUT_RDWR)
        except Exception as e:
            print("Error:", e)
        finally:
            try:
                ssock.close()
            except:
                pass
