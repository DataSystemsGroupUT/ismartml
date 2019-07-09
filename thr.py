import threading
import time

class Result_C:

    def __init__(self, value):
        self.value = value


#val=10
val=Result_C(10)


def f(val):
    for i in range(1, 10):
         time.sleep(1)
         print(i)
         #global val
         val.value*=i

def dd(val):
    for i in range(1, 5):
         time.sleep(2)
         print(val.value)


thread = threading.Thread(target=f,args=[val])
thread2 = threading.Thread(target=dd,args=[val])
thread.start()
thread2.start()

print("This may print while the thread is running.")
#thread.join()
print("This will always print after the thread has finished.")

