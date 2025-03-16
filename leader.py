import machine
import time
import ulab.numpy as np

spi = machine.SPI(0, baudrate=1000000, polarity=0, phase=0)
cspin = [machine.Pin(5, machine.Pin.OUT), 
           machine.Pin(6, machine.Pin.OUT), 
           machine.Pin(7, machine.Pin.OUT), 
           machine.Pin(8, machine.Pin.OUT)]

for i in cspin:
    i.value(1)

n_h, n_x = 10, 5
h_past = np.random.randn(n_h, 1)
x_t = np.random.randn(n_x, 1)
c_past = np.random.randn(n_h, 1)
hehe = np.vstack((h_past, x_t))

results = []
for i in range(4):
    cspin[i].value(0)
    time.sleep(0.01)
    send_str = ",".join(map(str, hehe.flatten()))
    spi.write(send_str.encode())
    response = spi.read(128)
    if response:
        results.append(np.array([float(x) for x in response.decode().strip().split(",")]).reshape(n_h, 1))
    cspin[i].value(1)
    time.sleep(0.01)

f_t, u_t, headc_t, o_t = results
f_t = 1 / (1 + np.exp(-f_t))
u_t = 1 / (1 + np.exp(-u_t))
o_t = 1 / (1 + np.exp(-o_t))
headc_t = (np.exp(headc_t) - np.exp(-headc_t)) / (np.exp(headc_t) + np.exp(-headc_t))
c_t = f_t * c_past + u_t * headc_t
h_t = o_t * (np.exp(c_t) - np.exp(-c_t)) / (np.exp(c_t) + np.exp(-c_t))
#순전파 END;

