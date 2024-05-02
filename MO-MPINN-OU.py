#  Description: The MO-MPINN for solving two-dimensional DR-PDEE of 
#  eight-dimensional Ornstein-Uhlenbeck process with random stiffness
#  ========================================================
#  ====        Authors:         	Teng-Teng Hao   =======
#  ====        Creation Date:   	2024/05/02      =======
#  ====        Modfication Date:	2024/05/02      =======
#  ========================================================
#
# input--------------------------------------------------------------------








import tensorflow as tf
import scipy.optimize
import scipy.io
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.compat.v1.Session(config=config)

# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

# Get the current working directory folder path
path = os.getcwd()
name = 'OU_0.5_10s.mat'
file_path = os.path.join(path, name)
para = loadmat(file_path)

D = para["D"]
T = para["tpre"]
t = para["t"][0, 1:]
dt = para["dt"]
xl = para["xl"]
Xl = para["Xl"]
Vl = para["Vl"]
XTpred = para["XTpred"]

alpha = para["alpha"]
Neq = para["Neq"]
r = int(alpha * 2.0 * Neq)

x_ic_train = para["xt_ic"]
y_ic_train = para["y_ic"]
x_bc_train = para["xt_bc"]

y_low, y_up = xl.min(), xl.max()
y0, t0, t_end = 0, T.min(), T.max()
x_p_low, x_p_up = np.hstack([y_low, t0]), np.hstack([y_up, t_end])
" --------------------------------------------Function Definition---------------------------------------------"


def generator_x_for_y(Num_x):
    np.random.seed(1111)
    xt_lhs = x_p_low.reshape(1, -1) + lhs(2, Num_x) * (x_p_up - x_p_low)
    return xt_lhs


def lowess(x):
    x_f = x[:, 0:1]
    num_x = x[:, 0].size
    t_indices = np.floor(x[:, 1] / dt + 0.00000001).astype(int)
    times = x[:, 1] / dt - t_indices
    x_data0 = np.vstack([Xl[:, i - 1] for i in t_indices])
    v_data0 = np.vstack([Vl[:, i - 1] for i in t_indices])
    x_data1 = np.vstack([Xl[:, i] for i in t_indices])
    v_data1 = np.vstack([Vl[:, i] for i in t_indices])
    x_data2 = x_data0 + (x_data1 - x_data0) * times
    v_data2 = v_data0 + (v_data1 - v_data0) * times
    x_a_data = np.vstack([x_data2, -x_data2]).T
    v_a_data = np.vstack([v_data2, -v_data2]).T
    sigmaX = np.std(x_a_data, axis=1).reshape(-1, 1) - np.zeros([num_x, int(2.0 * Neq)])

    d = tf.sqrt(((x_a_data - x_f) / sigmaX) ** 2)
    d_sort = tf.sort(d, axis=1)
    h = tf.reshape(d_sort[:, r - 1], (-1, 1))
    dis = d / h
    one_indices = tf.where(tf.equal(dis, 1))
    dis = tf.tensor_scatter_nd_update(dis, one_indices, tf.ones(len(one_indices), dtype=d.dtype) - 0.0000000001)
    lw = tf.maximum(0, tf.sign(1 - dis)) * (1 - dis ** 3) ** 3
    zero_indices = tf.where(tf.not_equal(lw, 0))
    lw1 = tf.reshape(tf.gather_nd(lw, zero_indices), (num_x, r))
    x_data = tf.reshape(tf.gather_nd(x_a_data, zero_indices), (num_x, r))
    v_data = tf.reshape(tf.gather_nd(v_a_data, zero_indices), (num_x, r))
    w0 = np.sum(lw1, axis=1).reshape(-1, 1)
    wx = np.sum(lw1 * x_data, axis=1).reshape(-1, 1)
    wv = np.sum(lw1 * v_data, axis=1).reshape(-1, 1)
    wxx = np.sum(lw1 * x_data * x_data, axis=1).reshape(-1, 1)
    wxv = np.sum(lw1 * x_data * v_data, axis=1).reshape(-1, 1)
    D0 = w0 * wxx - wx ** 2
    D1 = wxx * wv - wx * wxv
    D2 = w0 * wxv - wx * wv
    Beta = np.hstack([D1, D2]) / D0
    Aint = Beta[:, 0:1] + Beta[:, 1:2] * x_f
    return Aint


" --------------------------------------------MO-MPINN---------------------------------------------"


class MlpPINN(tf.Module):
    def __init__(self, layers_a, layers_p):
        self.step_lbfgs = 0
        self.W = []  # Weights and biases
        self.parameters = 0  # total number of parameters
        self.layers_a = layers_a
        self.layers_p = layers_p
        self.n_layers = len(self.layers_a) - 1 + (len(self.layers_p) - 1)
        self.W1 = 2 * len(self.layers_a) - 2
        self.jj = 0
        for i in range(self.n_layers):
            if i < (len(self.layers_a) - 1):
                input_dim = self.layers_a[i]
                output_dim = self.layers_a[i + 1]
                std_dv = np.sqrt((2.0 / (input_dim + output_dim)))  # Xavier standard deviation
                w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
                w = tf.Variable(w, trainable=True, name='w' + str(i + 1))
                b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True, name='b' + str(i + 1))
                self.parameters += input_dim * output_dim + output_dim
                self.W.append(w)
                self.W.append(b)
            else:
                input_dim = self.layers_p[self.jj]
                output_dim = self.layers_p[self.jj + 1]
                std_dv = np.sqrt((2.0 / (input_dim + output_dim)))
                w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
                w = tf.Variable(w, trainable=True, name='w' + str(i + 1))
                b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True,
                                name='b' + str(i + 1))
                self.parameters += input_dim * output_dim + output_dim
                self.W.append(w)
                self.W.append(b)
                self.jj += 1

    @tf.function
    def net_a(self, x):
        layers = self.layers_a
        weights = self.W[0:self.W1]
        a = x
        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, weights[2 * i]), weights[2 * i + 1])
            a = tf.nn.tanh(z)
        i = len(layers) - 2
        z = tf.add(tf.matmul(a, weights[2 * i]), weights[2 * i + 1])
        return z

    @tf.function
    def net_p(self, x):
        layers = self.layers_p
        weights = self.W[self.W1:]
        a = x
        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, weights[2 * i]), weights[2 * i + 1])
            a = tf.nn.tanh(z)
        i = len(layers) - 2
        z = tf.add(tf.matmul(a, weights[2 * i]), weights[2 * i + 1])
        a = z ** 4
        return a

    @tf.function
    def evaluate(self, x):
        return tf.stack([self.net_a(x), self.net_p(x)], axis=1)

    @tf.function
    def get_weights(self):
        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        for i in range(self.n_layers):
            w_1d = tf.reshape(self.W[2 * i], [-1])  # flatten weights
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])  # flatten biases
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)  # concat weights
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)  # concat biases
        return parameters_1d

    @tf.function
    def set_weights(self, w_matrix):
        for i in range(self.n_layers):
            shape_w = tf.shape(self.W[2 * i])  # shape of the weight tensor
            size_w = tf.size(self.W[2 * i])  # size of the weight tensor
            shape_b = tf.shape(self.W[2 * i + 1])  # shape of the bias tensor
            size_b = tf.size(self.W[2 * i + 1])  # size of the bias tensor
            pick_w = tf.slice(w_matrix, [0], [size_w])  # pick the weights
            self.W[2 * i].assign(tf.reshape(pick_w, shape_w))  # assign weights
            w_matrix = tf.slice(w_matrix, [size_w], [-1])  # remove used weights from matrix
            pick_b = tf.slice(w_matrix, [0], [size_b])  # pick the biases
            self.W[2 * i + 1].assign(tf.reshape(pick_b, shape_b))  # assign biases
            w_matrix = tf.slice(w_matrix, [size_b], [-1])  # remove used biases from matrix

    @tf.function
    def loss_ics(self, x_ic, y_ic):
        return tf.reduce_mean(tf.square(y_ic - self.net_p(x_ic)))

    @tf.function
    def loss_bcs(self, x_bc):
        return tf.reduce_mean(tf.square(self.net_p(x_bc)))

    @tf.function
    def loss_res(self, x_to_train_f):
        x_f = x_to_train_f[:, 0:1]
        t_f = x_to_train_f[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)
            g = tf.stack([x_f[:, 0], t_f[:, 0]], axis=1)
            z1 = self.net_a(g)
            z2 = self.net_p(g)
            z3 = z1 * z2
            u_x = tape.gradient(z2, x_f)
        u_t = tape.gradient(z2, t_f)
        ua_x = tape.gradient(z3, x_f)
        u_xx = tape.gradient(u_x, x_f)
        del tape
        res = u_t + ua_x - D / 2 * u_xx
        return res

    @tf.function
    def loss_pde(self, x):
        return tf.reduce_mean(tf.square(self.loss_res(x)))

    @tf.function
    def loss_a(self, x):
        return tf.reduce_mean(tf.square(self.net_a(x) - Ain_f))

    @tf.function
    def loss(self, x_ic, y_ic, x_bc, x_f):
        loss = Weights[0] * self.loss_ics(x_ic, y_ic) + Weights[1] * self.loss_bcs(x_bc) + \
               Weights[2] * self.loss_pde(x_f) + Weights[3] * self.loss_a(x_f)
        return loss

    @tf.function
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val = self.loss(x_ic_train, y_ic_train, x_bc_train, x_f_train)
        grads = tape.gradient(loss_val, self.trainable_variables)
        del tape
        grads_1d = []  # flatten grads
        for i in range(self.n_layers):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])  # flatten weights
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])  # flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)  # concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)  # concat grad_biases
        return loss_val, grads_1d

    # @tf.function
    def optimizer_callback(self, parameters):
        self.step_lbfgs += 1
        if self.step_lbfgs % 100 == 0:
            loss_all.append(self.loss(x_ic_train, y_ic_train, x_bc_train, x_f_train))
            loss_ics.append(self.loss_ics(x_ic_train, y_ic_train))
            loss_bcs.append(self.loss_bcs(x_bc_train))
            loss_res.append(self.loss_pde(x_f_train))
            loss_a.append(self.loss_a(x_f_train))
            Elapsed = time.time() - start_time1
            U_pred = np.reshape(self.evaluate(XTpred), (-1, 2), order='F')
            Loss_matrix = np.stack([loss_all, loss_ics, loss_bcs, loss_res, loss_a], axis=0).T
            savemat('loss_matrix.mat', mdict={'loss_matrix': Loss_matrix})
            savemat('u_pred.mat', mdict={'u_pred': U_pred})
            tf.print(self.step_lbfgs, Elapsed, loss_all[-1], loss_ics[-1], loss_bcs[-1], loss_res[-1], loss_a[-1])

    @tf.function
    def adaptive_gradients(self):
        with tf.GradientTape() as tape:
            tape.watch(self.W)
            loss_val = self.loss(x_ic_train, y_ic_train, x_bc_train, x_f_train)
        grads = tape.gradient(loss_val, self.W)
        del tape
        return loss_val, grads


"--------------------------------------------Data generation, Parameter setting, Training---------------------------------------------"
start_time0 = time.time()

N_f = 50000
x_f_train = generator_x_for_y(N_f)
Ain_f = lowess(x_f_train)

step_Pt, num_epochs, LBFGS_maxiter = 0, 5000, 20000
Weights = [1000.0, 1.0, 1000.0, 1.0]
loss_ics, loss_bcs, loss_res, loss_all, loss_a = [], [], [], [], []

layers_aint = [2] + [10] * 5 + [1]  # NN for aint
layers_pdf = [2] + [10] * 5 + [1]  # NN for pdf
mompinn = MlpPINN(layers_aint, layers_pdf)
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
for epoch in range(num_epochs + 1):
    Loss_val, Grads = mompinn.adaptive_gradients()
    Optimizer.apply_gradients(zip(Grads, mompinn.W))  # gradient descent weights
    step_Pt += 1
    if step_Pt % 100 == 0:
        loss_all.append(Loss_val)
        loss_ics.append(mompinn.loss_ics(x_ic_train, y_ic_train))
        loss_bcs.append(mompinn.loss_bcs(x_bc_train))
        loss_res.append(mompinn.loss_pde(x_f_train))
        loss_a.append(mompinn.loss_a(x_f_train))
        u_pred = np.reshape(mompinn.evaluate(XTpred), (-1, 2), order='F')
        savemat('u_pred.mat', mdict={'u_pred': u_pred})
        para_mompinn = mompinn.get_weights().numpy()
        savemat('para_mompinn.mat', mdict={'para_mompinn': para_mompinn})
        loss_matrix = np.stack([loss_all, loss_ics, loss_bcs, loss_res, loss_a], axis=0).T
        savemat('loss_matrix.mat', mdict={'loss_matrix': loss_matrix})
        dur, prog = time.time() - start_time0, (epoch + 1) * 100 // num_epochs
        print(
            "\rTraining progress: {:d}% ({:d}/{:d}) {:.2f}s loss={:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                prog, epoch + 1, num_epochs, dur,
                loss_all[-1], loss_ics[-1], loss_bcs[-1], loss_res[-1], loss_a[-1]), end="")

time_adam = time.time() - start_time0
print("\n" + "Adam optimization execution is complete, go to the next step".center(100 // 2, "-"))
print('Time_Adam: %.2fs' % time_adam)

start_time1 = time.time()
init_params = mompinn.get_weights().numpy()
results = scipy.optimize.minimize(fun=mompinn.optimizerfunc,
                                  x0=init_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac=True,
                                  callback=mompinn.optimizer_callback,
                                  options={'disp': None,
                                           'maxcor': 20,
                                           'ftol': 1e-8 * np.finfo(float).eps,
                                           'gtol': 1e-15,
                                           'maxfun': 50000,
                                           'maxiter': LBFGS_maxiter,
                                           'iprint': -1,
                                           'maxls': 50})
para_mompinn = mompinn.get_weights().numpy()
time_lbfgs = time.time() - start_time1

start_time2 = time.time()
u_pred = np.reshape(mompinn.evaluate(XTpred), (-1, 2), order='F')
time_pred = time.time() - start_time2
time_all = time.time() - start_time0
loss_matrix = np.stack([loss_all, loss_ics, loss_bcs, loss_res, loss_a], axis=0).T
time_cost = np.stack([time_all, time_adam, time_lbfgs, time_pred], axis=0).T
savemat('mompinn_results.mat', {'u_pred': u_pred, 'loss_matrix': loss_matrix, 'time_cost': time_cost,
                                'para_mompinn': para_mompinn})

print('Time_adam: %.2fs' % time_adam)
print('time_lbfgs: %.2fs' % time_lbfgs)
print('time_pred: %.2fs' % time_pred)
print('time_all: %.2fs' % time_all)
