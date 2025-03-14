import machine
import ulab.numpy as np

spi = machine.SPI(0, baudrate=1000000, polarity=0, phase=0)
cs = machine.Pin(5, machine.Pin.IN, machine.Pin.PULL_UP)

n_h = 10

n_x =5
W_f = np.ones((n_h, n_x + n_h))
b_f = np.zeros((n_h, 1))

while True:
    if cs.value()==0:
        tmpdata = spi.read(128)
        if tmpdata:
            gotvalue = np.array([float(x) for x in tmpdata.decode().strip().split(",")]).reshape(-1, 1)
            res = np.dot(W_f, gotvalue)+b_f
            sending = ",".join(map(str, res.flatten()))
            spi.write(sending.encode())
