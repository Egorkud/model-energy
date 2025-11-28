## Installation
### Install in system (Linux)
sudo snap install astral-uv --classic

### Use in project (.venv will be created automatically)
uv sync


# TASK 2 commands
```shell
mkdir -p ~/project_tls && cd ~/project_tls
```

```shell
openssl req -x509 -nodes -newkey rsa:4096 -days 365 \
  -keyout server.key -out server.crt \
  -subj "/C=UA/ST=Kyiv/L=Kyiv/O=MyLab/CN=localhost"
```

```shell
cat server.key server.crt > server_full.pem
```

```shell
openssl x509 -in server.crt -noout -text
openssl x509 -in server.crt -noout -fingerprint -sha256
```

