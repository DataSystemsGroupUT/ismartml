from hashlib import md5
#print(hashlib.md5(file_as_bytes(open(full_path, 'rb'))).hexdigest())


#file_as_bytes("boston.npy")
def hash_file(path):
    chk= md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            chk.update(chunk)
    return chk.hexdigest()
if __name__=="__main__":
    print(hash_file("boston.npy"))
    print(hash_file("reg.py"))
    print(hash_file("cat.txt"))
