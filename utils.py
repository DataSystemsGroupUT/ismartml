from hashlib import md5

def hash_file(path):
    chk= md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            chk.update(chunk)
    return chk.hexdigest()

