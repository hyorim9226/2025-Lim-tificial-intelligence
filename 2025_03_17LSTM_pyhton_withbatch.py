# %%
import numpy as np

# %%
class LSTM:

    def __init__(self, n_x, n_h, n_y):
        self.n_h = n_h
        self.n_x = n_x
        self.n_y = n_y
        self.W_f = np.random.randn(n_h, n_h + n_x) * np.sqrt(2/(n_h+n_x))
        self.W_u = np.random.randn(n_h, n_h + n_x) * np.sqrt(2/(n_h+n_x))
        self.W_o = np.random.randn(n_h, n_h + n_x) * np.sqrt(2/(n_h+n_x))
        self.W_c = np.random.randn(n_h, n_h + n_x) * np.sqrt(2/(n_h+n_x))
       
        self.b_f = np.zeros((n_h, 1)) * 0.01
        self.b_u = np.zeros((n_h, 1)) * 0.01
        self.b_o = np.zeros((n_h, 1)) * 0.01
        self.b_c = np.zeros((n_h, 1)) * 0.01

        self.W_y = np.random.randn(n_y, n_h) * 0.01
        self.b_y = np.zeros((n_y, 1))
        self.cache = None
        self.grads = [np.zeros_like(self.W_f), np.zeros_like(self.W_u), np.zeros_like(self.W_o), np.zeros_like(self.W_c), np.zeros_like(self.b_f), np.zeros_like(self.b_u), np.zeros_like(self.b_o), np.zeros_like(self.b_c), np.zeros_like(self.W_y), np.zeros_like(self.b_y)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, h_past, x_t, c_past):
        batch_size = x_t.shape[1]  
        hehe = np.vstack((h_past, x_t))  

        f_t = self.sigmoid(np.dot(self.W_f, hehe) + self.b_f)
        u_t = self.sigmoid(np.dot(self.W_u, hehe) + self.b_u)
        o_t = self.sigmoid(np.dot(self.W_o, hehe) + self.b_o)
        headc_t = self.tanh(np.dot(self.W_c, hehe) + self.b_c)

        c_t = f_t * c_past + u_t * headc_t  
        h_t = o_t * self.tanh(c_t)  

        z_t = np.dot(self.W_y, h_t) + self.b_y
        y_hat = self.softmax(z_t)  

        self.cache = (x_t, h_past, c_past, f_t, u_t, o_t, headc_t, c_t, h_t, y_hat)

        return h_t, c_t, y_hat, z_t

    def backward(self, dh_next, dc_next, y_true):
        x_t, h_past, c_past, f_t, u_t, o_t, headc_t, c_next, h_t, y_hat = self.cache
        tanc_next = self.tanh(c_next)

        
        dz_t = y_hat - y_true
        dW_y = np.dot(dz_t, h_t.T)
        db_y = np.sum(dz_t, axis=1, keepdims=True)

        ds = dh_next * o_t * (1 - tanc_next**2) + dc_next
        dc_past = ds * f_t

        df = ds * c_past * f_t * (1 - f_t)
        du = ds * headc_t * u_t * (1 - u_t)
        do = dh_next * tanc_next * o_t * (1 - o_t)
        dc = ds * u_t * (1 - headc_t**2)

        hehe = np.vstack((h_past, x_t))
        dW_f = np.dot(df, hehe.T)
        dW_u = np.dot(du, hehe.T)
        dW_o = np.dot(do, hehe.T)
        dW_c = np.dot(dc, hehe.T)
        db_f = np.sum(df, axis=1, keepdims=True)
        db_u = np.sum(du, axis=1, keepdims=True)
        db_o = np.sum(do, axis=1, keepdims=True)
        db_c = np.sum(dc, axis=1, keepdims=True)
        dh_past = np.dot(self.W_f[:, :self.n_h].T, df) + np.dot(self.W_u[:, :self.n_h].T, du) + np.dot(self.W_o[:, :self.n_h].T, do) + np.dot(self.W_c[:, :self.n_h].T, dc) + np.dot(self.W_y.T, dz_t)

        self.grads[0] = dW_f
        self.grads[1] = dW_u
        self.grads[2] = dW_o
        self.grads[3] = dW_c
        self.grads[4] = db_f
        self.grads[5] = db_u
        self.grads[6] = db_o
        self.grads[7] = db_c
        self.grads[8] = dW_y
        self.grads[9] = db_y

        return dh_past, dc_past

    def update(self, learning_rate):
        for i in range(len(self.grads)):
            self.grads[i] *= (1 / batch_size)  
            self.grads[i] -= learning_rate * self.grads[i]  

# %%
n_x = 5
n_h = 10
batch_size = 5 

lstm = LSTM(n_x=n_x, n_h=n_h, n_y=1)
# %%
x = np.array([[-0.0495535, 0.2008880, 0.2930990, -0.0285330, -0.2115030],
    [-0.2336310, 0.0189126, -0.1846580, -0.1353680, 0.2942170],
    [-0.0225936, 0.1517270, -0.0345324, 0.1838040, 0.2126690],
    [-0.1213360, 0.1562240, 0.0415218, 0.2318500, -0.0068506],
    [0.0852375, 0.1702850, 0.1999780, -0.1927900, -0.0141844],
    [-0.2036370, -0.0995580, -0.1636710, -0.0066911, 0.0156342],
    [-0.1126930, 0.1084010, -0.1617170, 0.2913920, -0.1507980],
    [-0.0148840, 0.1612430, -0.1878050, -0.2219460, 0.1410460],
    [0.1198300, -0.2329980, 0.0452951, 0.1705200, -0.1988260],
    [0.0104172, -0.2468400, 0.2511100, 0.2335360, 0.2187400],
    [-0.0765870, -0.1110770, 0.0628817, 0.1433000, -0.2045370],
    [-0.0353429, -0.2008080, -0.1689490, 0.1952930, 0.1812200],
    [-0.0194253, 0.1999720, -0.2106910, 0.2483420, 0.1947920],
    [-0.0572571, -0.2077380, -0.0178791, -0.2957700, 0.1409140],
    [-0.0269522, -0.1855840, -0.1874070, 0.2928830, -0.1490250],
    [0.2905000, -0.0537734, 0.0804795, -0.0035505, 0.0713696],
    [-0.2080550, 0.2049740, -0.2727730, 0.2810620, -0.2369590],
    [-0.2734300, 0.0971220, -0.0548890, 0.0856415, -0.2318550],
    [-0.0296142, -0.2229130, -0.2095950, -0.1489230, 0.0134664],
    [-0.2920470, -0.1361660, -0.0690835, 0.0368628, 0.1228950],
    [0.6804980, 0.5838080, 0.7363420, 0.7893920, 0.5888150],
    [0.6367920, 0.9786730, 0.9702610, 0.6410970, 0.7026760],
    [0.6011990, 0.9241290, 0.8888960, 0.6931200, 0.9682510],
    [0.5314040, 0.6117050, 0.9861350, 0.6836870, 0.6657360],
    [0.6163250, 0.7980310, 0.9972130, 0.7457960, 0.5552360],
    [0.6912060, 0.9130120, 0.8203070, 0.9975940, 0.7615620],
    [0.8322720, 0.6684010, 0.7474360, 0.6535700, 0.5969570],
    [0.5614510, 0.7930520, 0.5127360, 0.5828140, 0.6941610],
    [0.7063490, 0.8334810, 0.7595500, 0.7282120, 0.5309490],
    [0.8418380, 0.8385730, 0.8016880, 0.8915860, 0.5478830],
    [0.7109540, 0.7822390, 0.5885250, 0.9767660, 0.5967980],
    [0.9030160, 0.9540120, 0.5645290, 0.9415510, 0.5959370],
    [0.8513460, 0.8262000, 0.7288070, 0.9709240, 0.9516100],
    [0.7436080, 0.9588600, 0.6425340, 0.7704930, 0.8489160],
    [0.8844600, 0.7947130, 0.6567900, 0.7212200, 0.7752380],
    [0.5495960, 0.7151610, 0.9074890, 0.7246360, 0.8053240],
    [0.8001500, 0.7286490, 0.7574770, 0.5746680, 0.7909890],
    [0.5456610, 0.5993310, 0.8550080, 0.8867530, 0.5073650],
    [0.7398610, 0.9435000, 0.6897520, 0.5717600, 0.5665310],
    [0.7948440, 0.8538850, 0.9455280, 0.8411180, 0.5138040]]
   )
# %%
epoch = 20
h_t = np.zeros((n_h, batch_size))  
c_t = np.zeros((n_h, batch_size))
learning_rate = 0.01 #이거 알아서 설정해야하는것같은데 우예하는지 모르겟노
y = np.array([1 for i in range(20)] + [0 for _ in range(20)])
for i in range(epoch):
    batch_loss = 0
    for j in range(0, len(x), batch_size):
        x_batch = x[j:j+batch_size].reshape(-1, batch_size)  
        y_true_batch = y[j:j+batch_size].reshape(1, -1) 
        print(y_true_batch)
        
        h_t, c_t, y_hat, _ = lstm.forward(h_t, x_batch, c_t)
        print(y_hat)
        loss = -np.sum(np.log(y_hat+ 1e-8) * y_true_batch)  
        batch_loss += loss
        print(loss)
        dh_next = np.zeros_like(h_t)
        dc_next = np.zeros_like(c_t)

        dh_next, dc_next = lstm.backward(dh_next, dc_next, y_true_batch)
        lstm.update(learning_rate)

    print(f"Epoch {i+1}, Loss: {batch_loss / (len(x) // batch_size)}")

# %%
