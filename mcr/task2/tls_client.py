import socket, ssl

HOST = '127.0.0.1'
PORT = 4443
CAFILE = '/home/vboxuser/project_tls/server.crt'   # використовуємо самопідписний сертифікат як CA

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=CAFILE)
context.check_hostname = False   # сервер localhost, CN=localhost - можна True якщо хочеш перевіряти
context.verify_mode = ssl.CERT_REQUIRED

with socket.create_connection((HOST, PORT)) as sock:
    with context.wrap_socket(sock, server_hostname='localhost') as ssock:
        print("Connected. TLS version:", ssock.version())
        print("Cipher:", ssock.cipher())
        # відправимо простий HTTP GET
        ssock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        resp = ssock.recv(4096)
        print("Server response (truncated):", resp[:200])
