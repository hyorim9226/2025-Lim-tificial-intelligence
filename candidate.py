import machine
import ulab.numpy as np

spi = machine.SPI(0, baudrate=1000000, polarity=0, phase=0)
cs = machine.Pin(7, machine.Pin.IN, machine.Pin.PULL_UP)

n_h = 10
n_x =5
W_c = np.ones((n_h, n_x + n_h))
b_c = np.zeros((n_h, 1))

while True:
    if cs.value() == 0:
        data = spi.read(128)
        if data:
            receive = np.array([float(x) for x in data.decode().strip().split(",")]).reshape(-1, 1)
            rees = np.dot(W_c, receive) + b_c
            sending = ",".join(map(str, rees.flatten()))
            spi.write(sending.encode())
