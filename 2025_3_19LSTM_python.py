# %%
import numpy as np

# %%
class Cell:
    def __init__(self):
        self.W_f = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_u = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_o = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_c = np.random.randn(1, 2) * np.sqrt(1/2)
        
        self.b_f = np.random.randn(1, 1)
        self.b_u = np.random.randn(1, 1)
        self.b_o = np.random.randn(1, 1)
        self.b_c = np.random.randn(1, 1)

        self.cache = None
        self.grads = []

    def sigmoid(self, n):
        return 1/(1+np.exp(-n))
    
    def psigmoid(self, n):
        n = self.sigmoid(n)
        return n * (1 - n)
    
    def tanh(self, n):
        return np.tanh(n)
    
    def softmax(self, n):
        exp_n = np.exp(n - np.max(n))
        return exp_n / np.sum(exp_n, axis=0, keepdims=True)
    
    def forward(self, x_t, t):
        global h_t, c_t

        hehe = np.vstack((h_t[t - 1], x_t))

        gamma_f = np.dot(self.W_f, hehe) + self.b_f
        f_t = self.sigmoid(gamma_f)

        gamma_u = np.dot(self.W_u, hehe) + self.b_u
        u_t = self.sigmoid(gamma_u)
        
        gamma_o = np.dot(self.W_o, hehe) + self.b_o
        o_t = self.sigmoid(gamma_o)

        gamma_c = np.dot(self.W_c, hehe) + self.b_c
        headc_t = self.tanh(gamma_c)

        c_t[t] = f_t * c_t[t - 1] + u_t * headc_t # 이게 c_next로 cache에 저장
        h_t[t] = o_t * self.tanh(c_t[t]) # 이것도 h_next로 cache에 저장

        self.cache = (gamma_f, gamma_u, gamma_o, gamma_c, f_t, u_t, o_t, headc_t)

        return None

    def backward(self, dh, t): #dh = dL/dh_t
        global h_t, c_t
        gamma_f, gamma_u, gamma_o, gamma_c, f_t, u_t, o_t, headc_t = self.cache
        
        dw_o = dh * self.tanh(c_t[t]) * self.psigmoid(gamma_o) * h_t[t - 1]
        db_o = dh * self.tanh(c_t[t]) * self.psigmoid(gamma_o)

        dw_f = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * c_t[t - 1] * self.psigmoid(gamma_f) * h_t[t - 1]
        db_f = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * c_t[t - 1] * self.psigmoid(gamma_f)

        dw_c = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * u_t * (1 - self.tanh(headc_t) ** 2) * h_t[t - 1]
        db_c = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * u_t * (1 - self.tanh(headc_t) ** 2)

        dw_u = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * headc_t * self.psigmoid(gamma_u) * h_t[t - 1]
        db_u = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * headc_t * self.psigmoid(gamma_u)

        dh_next = self.psigmoid(gamma_o) * self.W_o[0, 0] * self.tanh(c_t[t]) \
                + o_t * (1 - self.tanh(c_t[t]) ** 2) * self.psigmoid(gamma_f) * self.W_f[0, 0] * c_t[t - 1] \
                + self.psigmoid(gamma_u) * self.W_u[0, 0] + u_t * (1 - self.tanh(gamma_c) ** 2) * self.W_c[0, 0]

        self.grads = [dw_f.item(), dw_u.item(), dw_o.item(), dw_c.item(), db_f.item(), db_u.item(), db_o.item(), db_c.item()]   
        return dh * dh_next
    
    def update(self, learning_rate):
        self.W_c[0, 0] = self.W_c[0, 0] - learning_rate * self.grads[3]
        self.W_f[0, 0] = self.W_f[0, 0] - learning_rate * self.grads[0]
        self.W_o[0, 0] = self.W_o[0, 0] - learning_rate * self.grads[2]
        self.W_u[0, 0] = self.W_u[0, 0] - learning_rate * self.grads[1]
        self.b_c[0] = self.b_c[0] - learning_rate * self.grads[7]
        self.b_f[0] = self.b_f[0] - learning_rate * self.grads[4]
        self.b_o[0] = self.b_o[0] - learning_rate * self.grads[6]
        self.b_u[0] = self.b_u[0] - learning_rate * self.grads[5]

#%%
class Z:
    def __init__(self):
        self.W_f = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_u = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_o = np.random.randn(1, 2) * np.sqrt(1/2)
        self.W_c = np.random.randn(1, 2) * np.sqrt(1/2)
        
        self.b_f = np.random.randn(1, 1) * 0.01
        self.b_u = np.random.randn(1, 1) * 0.01
        self.b_o = np.random.randn(1, 1) * 0.01
        self.b_c = np.random.randn(1, 1) * 0.01

        self.w_z = np.random.randn(1, 1)
        self.b_z = np.random.randn(1, 1) * 0.01

        self.cache = None
        self.grads = []

    def sigmoid(self, n):
        return 1/(1+np.exp(-n))
    
    def psigmoid(self, n):
        n = self.sigmoid(n)
        return n * (1 - n)
    
    def tanh(self, n):
        return np.tanh(n)
    
    def softmax(self, n):
        exp_n = np.exp(n - np.max(n))
        return exp_n / np.sum(exp_n, axis=0, keepdims=True)
    
    def forward(self, x_t, t):
        global h_t, c_t

        hehe = np.vstack((h_t[t - 1], x_t))

        gamma_f = np.dot(self.W_f, hehe) + self.b_f
        f_t = self.sigmoid(gamma_f)

        gamma_u = np.dot(self.W_u, hehe) + self.b_u
        u_t = self.sigmoid(gamma_u)
        
        gamma_o = np.dot(self.W_o, hehe) + self.b_o
        o_t = self.sigmoid(gamma_o)

        gamma_c = np.dot(self.W_c, hehe) + self.b_c
        headc_t = self.tanh(gamma_c)

        c_t[t] = f_t * c_t[t - 1] + u_t * headc_t # 이게 c_next로 cache에 저장
        h_t[t] = o_t * self.tanh(c_t[t]) # 이것도 h_next로 cache에 저장

        z_t = np.dot(self.w_z, h_t[t]) + self.b_z
        y_hat = self.sigmoid(z_t)

        self.cache = (gamma_f, gamma_u, gamma_o, gamma_c, f_t, u_t, o_t, headc_t)

        return y_hat

    def backward(self, dh, t): #dh = dL/dz
        global h_t, c_t
        gamma_f, gamma_u, gamma_o, gamma_c, f_t, u_t, o_t, headc_t = self.cache
        
        dw_z = dh * h_t[t]
        db_z = dh

        dh = dh * self.w_z #dh = dL/dh_t

        dw_o = dh * self.tanh(c_t[t]) * self.psigmoid(gamma_o) * h_t[t - 1]
        db_o = dh * self.tanh(c_t[t]) * self.psigmoid(gamma_o)

        dw_f = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * c_t[t - 1] * self.psigmoid(gamma_f) * h_t[t - 1]
        db_f = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * c_t[t - 1] * self.psigmoid(gamma_f)

        dw_c = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * u_t * (1 - self.tanh(headc_t) ** 2) * h_t[t - 1]
        db_c = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * u_t * (1 - self.tanh(headc_t) ** 2)

        dw_u = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * headc_t * self.psigmoid(gamma_u) * h_t[t - 1]
        db_u = dh * o_t * (1 - self.tanh(c_t[t]) ** 2) * headc_t * self.psigmoid(gamma_u)

        dh_next = self.psigmoid(gamma_o) * self.W_o[0, 0] * self.tanh(c_t[t]) \
                + o_t * (1 - self.tanh(c_t[t]) ** 2) * self.psigmoid(gamma_f) * self.W_f[0, 0] * c_t[t - 1] \
                + self.psigmoid(gamma_u) * self.W_u[0, 0] + u_t * (1 - self.tanh(gamma_c) ** 2) * self.W_c[0, 0]


        self.grads = [dw_f.item(), dw_u.item(), dw_o.item(), dw_c.item(), dw_z.item(), db_f.item(), db_u.item(), db_o.item(), db_c.item(), db_z.item()]   
        return dh * dh_next
    
    def update(self, learning_rate):
        self.W_c[0, 0] = self.W_c[0, 0] - learning_rate * self.grads[3]
        self.W_f[0, 0] = self.W_f[0, 0] - learning_rate * self.grads[0]
        self.W_o[0, 0] = self.W_o[0, 0] - learning_rate * self.grads[2]
        self.W_u[0, 0] = self.W_u[0, 0] - learning_rate * self.grads[1]
        self.w_z[0, 0] = self.w_z[0, 0] - learning_rate * self.grads[4]
        self.b_c = self.b_c - learning_rate * self.grads[8]
        self.b_f = self.b_f - learning_rate * self.grads[5]
        self.b_o = self.b_o - learning_rate * self.grads[7]
        self.b_u = self.b_u - learning_rate * self.grads[6]
        self.b_z = self.b_z - learning_rate * self.grads[9]
# %%

x = [
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0]
]

y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0]

# %%
T = len(x[-1])

#%%
epoch = 4000
learning_rate = 0.02
#%%
# lstm = [0] + [Cell() for i in range(T - 1)] + [Z()]
# h_t = [0 for i in range(T + 1)]
# c_t = [0 for i in range(T + 1)]

# for i in range(epoch):
#     loss = 0
#     cnt = 0
#     res = []
#     for j in range(len(x)):
#         for t in range(1, T):
#             lstm[t].forward(x[j][t - 1], t)
#         y_hat = lstm[-1].forward(x[j][-1], T)
    
#         if y_hat > 0.5 and y[j] == 1:
#             cnt += 1
#         elif y_hat <= 0.5 and y[j] == 0:
#             cnt += 1

#         loss += -(y[j] * np.log(y_hat) + (1 - y[j]) * np.log(1 - y_hat))
#         res.append([y[j], y_hat])

#         dh = lstm[-1].backward((y_hat - y[j]), T)
#         lstm[-1].update(learning_rate=learning_rate)

#         for t in range(T - 1, 0, -1):
#             dh = lstm[t].backward(dh, t)
#             lstm[t].update(learning_rate=learning_rate)
#             #print(dh)

#     print(f"epoch: {i}    loss: {loss/len(x)}   acc: {cnt/len(x) * 100}%")
#%%
lstm = [0] + [Cell() for i in range(T - 1)] + [Z()]
h_t = [0 for i in range(T + 1)]
c_t = [0 for i in range(T + 1)]

epoch = 0
while True:
    epoch += 1
    loss = 0
    cnt = 0
    
    for j in range(len(x)):
        for t in range(1, T):
            lstm[t].forward(x[j][t - 1], t)
        y_hat = lstm[-1].forward(x[j][-1], T)
    
        if y_hat > 0.5 and y[j] == 1:
            cnt += 1
        elif y_hat <= 0.5 and y[j] == 0:
            cnt += 1

        loss += -(y[j] * np.log(y_hat) + (1 - y[j]) * np.log(1 - y_hat))

        dh = lstm[-1].backward((y_hat - y[j]), T)
        lstm[-1].update(learning_rate=learning_rate)

        for t in range(T - 1, 0, -1):
            dh = lstm[t].backward(dh, t)
            lstm[t].update(learning_rate=learning_rate)
            #print(dh)
            
    print(f"epoch: {epoch}    loss: {loss/len(x)}   acc: {cnt/len(x) * 100}%")
    if cnt/len(x) >= 0.99:
        break

# %%
#making prediction
num = list(map(int, input().split()))

for t in range(1, T):
    lstm[t].forward(num[t - 1], t)
y_hat = lstm[-1].forward(num[-1], T)

if y_hat > 0.5:
    print("Over than 20!")
elif y_hat <= 0.5:
    print("Under than 20!")
# %%

