import ulab.numpy as np
from machine import Pin, SPI

spi = SPI(0, baudrate=1000000, polarity=0, phase=0)
cs_pins = [Pin(i, Pin.OUT) for i in range(5, 9)]  

class LSTMLeader:
    def __init__(self, n_h, n_x):
        self.n_h = n_h
        self.n_x = n_x
        self.W_y = np.random.rand(1, n_h) * 0.01
        self.b_y = np.zeros((1, 1))
        self.cache = None

    def Communication(self, data, worker_id):
        cs_pins[worker_id].value(0)  
        spi.write(data)  
        result = spi.read(10)  
        cs_pins[worker_id].value(1)  
        return np.array(result)  

    def forward(self, h_past, x_t, c_past):
        hehe = np.vstack((h_past, x_t))  

        f_t = self.Communication(hehe.tobytes(), 0)  
        u_t = self.Communication(hehe.tobytes(), 1)
        headc_t = self.Communication(hehe.tobytes(), 2)
        o_t = self.Communication(hehe.tobytes(), 3)  

        c_t = f_t * c_past + u_t * headc_t
        h_t = o_t * np.tanh(c_t)

        z_t = np.dot(self.W_y, h_t) + self.b_y
        y_hat = np.exp(z_t) / np.sum(np.exp(z_t), axis=0)

        self.cache = (x_t, h_past, c_past, f_t, u_t, o_t, headc_t, c_t)
        return h_t, c_t, y_hat


n_x, n_h = 5, 10
lstm = LSTMLeader(n_h, n_x)
