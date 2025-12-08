import argparse
import binascii
import hashlib

from tqdm.contrib import tenumerate


def md5_chap(id_byte, password, challenge_bytes):
    m = hashlib.md5()
    m.update(bytes([id_byte]))
    m.update(password.encode('utf-8', errors='ignore'))
    m.update(challenge_bytes)
    return m.digest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', default='0x01')
    p.add_argument('--challenge', default='1234567890abcdef')
    p.add_argument('--response', default='518b84b3e32f8f8965762ded3b0d3fd0')

    p.add_argument('--wordlist', default='rockyou.txt')
    args = p.parse_args()

    id_byte = int(args.id, 16) if args.id.startswith('0x') else int(args.id)
    challenge = binascii.unhexlify(args.challenge)
    response = binascii.unhexlify(args.response)

    with open(args.wordlist, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in tenumerate(f, 1):
            pwd = line.rstrip('\n\r')
            if not pwd: continue
            if md5_chap(id_byte, pwd, challenge) == response:
                print(f'FOUND: {pwd}')
                return
    print('Not found')


if __name__ == "__main__":
    main()
