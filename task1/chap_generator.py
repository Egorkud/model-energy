import binascii
import hashlib

# HARD CODE
# Налаштування
PASSWORD = "sigma"  # Впишіть сюди будь-яке слово, яке Є в rockyou.txt
ID_HEX = "01"  # ID сесії (зазвичай 01)
CHALLENGE_HEX = "1234567890abcdef"  # Випадковий виклик

# Логіка (така сама, як у зломщику)
id_byte = bytes.fromhex(ID_HEX)
pwd_bytes = PASSWORD.encode('utf-8')
chal_bytes = binascii.unhexlify(CHALLENGE_HEX)

# MD5(ID + Password + Challenge)
m = hashlib.md5()
m.update(id_byte)
m.update(pwd_bytes)
m.update(chal_bytes)
response_hex = m.hexdigest()

print(f"--- ДАНІ ДЛЯ ВСТАВКИ ---")
print(f"Pass (для перевірки): {PASSWORD}")
print(f"p.add_argument('--id', default='0x{ID_HEX}')")
print(f"p.add_argument('--challenge', default='{CHALLENGE_HEX}')")
print(f"p.add_argument('--response', default='{response_hex}')")
