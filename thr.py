import threading
import time

val=10

def f():
    for i in range(1, 10):
         time.sleep(1)
         print(i)
         global val
         val=val*i

def dd():
    for i in range(1, 10):
         time.sleep(2)
         print(val)


thread = threading.Thread(target=f)
thread2 = threading.Thread(target=dd)
thread.start()
thread2.start()

print("This may print while the thread is running.")
#thread.join()
print("This will always print after the thread has finished.")

