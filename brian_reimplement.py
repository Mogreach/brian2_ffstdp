from brian2 import *
import brian2.numpy_ as np
import h5py
import time
import os

# set_device('cpp_standalone',build_on_run=False)
# set_device('runtime')
train_pics = 20000
test_pics = 100
test_interval = 1000
def save_script_as(new_filename):
    """
    将当前脚本另存为新文件。

    参数:
        new_filename (str): 新文件的路径。
    """
    # 获取当前脚本的文件路径
    current_script_path = os.path.abspath(__file__)

    # 读取当前脚本的内容
    with open(current_script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    # 将脚本内容保存到新文件
    with open(new_filename, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"脚本已另存为: {new_filename}")
def get_label_neg(label):
    # 生成0-10的所有可能值
    possible_values = np.arange(10)
    # 排除输入的值
    possible_values = possible_values[possible_values != label]
    fake_label = np.random.choice(possible_values)
    # 随机选择一个不同的值
    return fake_label
def overlay_label_on_img(img, label):
    # 找到数组的最大值
    max_value = np.max(img)
    # 将第一行第y列的值设置为最大值
    img[0, label] = max_value
    return img
def pos_derivative(x, theta):
    """
    计算 log(1 + exp(-x + theta)) 关于 x 的导数。

    参数:
        x (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = -1 / (1 + np.exp(x - theta))
    
    # 返回导数
    return sigmoid
def neg_derivative(y, theta):
    """
    计算 log(1 + exp(y - theta)) 关于 y 的导数。

    参数:
        y (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = 1 / (1 + np.exp(theta - y))
    
    # 返回导数
    return sigmoid
class SNN():
    def __init__(self):
        self.input_layer_size = 784
        self.hidden_layer_size = 784
        self.output_layer_size = 2
        self.cita_h = 1.6
        self.cita_o = 2.5
        self.reset = 0
        self.theta = 0.8
        # define the neuron model
        self.eqs = '''
                v:1
                cita_h:1
                cita_o:1
                reset:1
                '''
        self.base_frequency = 250
        self.learn_rate = 2 * 0.0078125
        self.time_step = 16
        self.sim_time = 16
        # make sure "test_steps * time_step = 20.0"
        self.test_steps = 1

        self.rates = zeros(self.input_layer_size)

        # input neurons
        inp = NeuronGroup(self.input_layer_size, 'v:1', method='exact', threshold='v>=1',
                          reset='v=0',name="input")
        def update_volt():
            inp.v += self.rates
        network_op = NetworkOperation(update_volt, dt=1.0 * ms)
        # hidden neurons
        hidden = NeuronGroup(self.hidden_layer_size, self.eqs, threshold="v>cita_h",
                             reset='v=reset', method="exact",name="hidden")
        hidden.cita_h = self.cita_h
        hidden.reset = self.reset
        output = NeuronGroup(self.output_layer_size, self.eqs, threshold='v>cita_o',
                             reset='v=reset', method='exact',name="output")
        output.cita_o = self.cita_o
        output.reset = self.reset


        conn_ih = Synapses(inp, hidden, model='w:1', on_pre='v_post += w',name="conn_ih")
        conn_ih.connect(p=1)
        conn_ih.w = np.random.randn(self.input_layer_size*self.hidden_layer_size)
        conn_ho = Synapses(hidden, output, model='w:1', on_pre='v_post += w',name="conn_ho")
        conn_ho.connect(p=1)
        conn_ho.w = np.random.randn(self.hidden_layer_size*self.output_layer_size)
        self.net = Network(conn_ih, conn_ho, network_op,
                           inp, hidden, output)
        # 定义脉冲变量
        self.spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        self.spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        self.spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')

        self.input_spike_count = array(self.spikemon_input.count).copy()
        self.hidden_spike_count = array(self.spikemon_hidden.count).copy()
        self.output_spike_count = array(self.spikemon_output.count).copy()
        # self.net.store("initial_weight")

    def set_input(self,img_array):
        self.rates = img_array / 255.0
        self.net.set_states({"input":{"v":zeros(self.input_layer_size)}})
        self.net.set_states({"hidden":{"v":zeros(self.hidden_layer_size)}})
        self.net.set_states({"output":{"v":zeros(self.output_layer_size)}})

    def goodness(self,spike):
        goodness = np.pow(spike/self.time_step,2)
        return self.time_step * goodness
    def weight_update(self, is_pos, spike_output, spike_input):
        freq = spike_output/ self.time_step
        goodness = self.goodness(spike_output)
        if is_pos:
            L_to_s_derivative = 2*freq*pos_derivative(goodness,self.theta)
            delta_weight = -1 * dot(spike_input.reshape((spike_input.size,1)),
                                L_to_s_derivative.reshape(1,spike_output.size)) * self.learn_rate
            # self.theta = self.theta - self.learn_rate * (L_to_s_derivative.mean() / 2)
        else:
            L_to_s_derivative =  2*freq*neg_derivative(goodness,self.theta)
            delta_weight = -1 * dot(spike_input.reshape((spike_input.size,1)),
                                L_to_s_derivative.reshape(1,spike_output.size)) * self.learn_rate
            # self.theta = self.theta - self.learn_rate * (L_to_s_derivative.mean() / 2)
        # weight_new = weight + delta_weight.flatten()
         # 限制权重范围在 [-1, 1] 之间
        # weight_new = np.clip(weight_new, -1, 1)
        return delta_weight.flatten()
        
    def train_step(self,is_pos,label):
        current_input_spike_count = self.spikemon_input.count
        current_hidden_spike_count = self.spikemon_hidden.count
        current_output_spike_count = self.spikemon_output.count
        self.net.run(self.time_step * ms)
        # time_step期间新增的脉冲数量：current_input - input （当前时刻 - 上一个时刻的累计脉冲数）
        spike_sum_input = subtract(current_input_spike_count , self.input_spike_count)
        spike_sum_hidden = subtract(current_hidden_spike_count , self.hidden_spike_count)
        spike_sum_output = subtract(current_output_spike_count , self.output_spike_count)
        # print("  [%2d] Spikes Sum:(%3d,%3d,%3d)" %
        #       (ct,sum(spike_sum_input), sum(spike_sum_hidden), sum(spike_sum_output)),end=", ")
        # print(spike_sum_output,end=", ")
        # 更新经过一个time_step后累计的脉冲数量
        self.input_spike_count = array(current_input_spike_count).copy()
        self.hidden_spike_count = array(current_hidden_spike_count).copy()
        self.output_spike_count = array(current_output_spike_count).copy()
        #---------------------------------------Output Layer Trained by LS Method--------------------------------------------------------------------
        ksai_output = zeros(self.output_layer_size)
        for _ in range(self.output_layer_size):
            if((spike_sum_output[_] >= 1) and (_ != label)):
                ksai_output[_] = -1
            if((spike_sum_output[_] == 0) and (_ == label)):
                ksai_output[_] = 1
        ho_delta_w = (dot(spike_sum_hidden.reshape((self.hidden_layer_size,1)),
                                ksai_output.reshape((1,self.output_layer_size))) * self.learn_rate).flatten()
        #---------------------------------------Output Layer Trained by LS Method--------------------------------------------------------------------
       
        # 计算梯度
        # ho_delta_w = self.weight_update(is_pos, spike_sum_output, spike_sum_hidden)
        ih_delta_w = self.weight_update(is_pos, spike_sum_hidden, spike_sum_input)
        # if is_pos:
        #     print(f"Pos_goodness:{self.goodness(spike_sum_hidden).mean()}")
        # else:
        #     print(f"Neg_goodness:{self.goodness(spike_sum_hidden).mean()}")
        return ho_delta_w, ih_delta_w, self.goodness(spike_sum_hidden).mean()

    def train(self, sim_time, img_array, label, is_pos):
        self.set_input(img_array)
        print("# Current label is : %d" % label)
        # analogy the input
        # rate = np.zeros(self.input_layer_size)
        self.spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        self.spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        self.spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')
        spikemon_list = [self.spikemon_input, self.spikemon_hidden, self.spikemon_output]
        self.net.add(spikemon_list)
        # 初始化在初始时刻脉冲放电次数
        self.input_spike_count = array(self.spikemon_input.count).copy()
        self.hidden_spike_count = array(self.spikemon_hidden.count).copy()
        self.output_spike_count = array(self.spikemon_output.count).copy()

        conn_ho_w = self.net.get_states()["conn_ho"]["w"]
        conn_ih_w = self.net.get_states()["conn_ih"]["w"]
        for ct in range(int(sim_time / self.time_step)):
            #--------------------------------FF_Method----------------------------------------------
            ho_delta_w, ih_delta_w, g = self.train_step(is_pos,label)
            # neg_ho_delta_w, neg_ih_delta_w  = self.train_step(img_neg_array,False,label)
            conn_ho_w = self.net.get_states()["conn_ho"]["w"]
            # update weights from input to hidden
            conn_ho_w = conn_ho_w + ho_delta_w
            # conn_ho_w = np.clip(conn_ho_w, -1, 1)
            self.net.set_states({"conn_ho": {"w": conn_ho_w}})

            conn_ih_w = self.net.get_states()["conn_ih"]["w"]
            conn_ih_w = conn_ih_w + ih_delta_w
            # conn_ih_w = np.clip(conn_ih_w, -1, 1)
            self.net.set_states({"conn_ih": {"w": conn_ih_w}})
            #--------------------------------LS_Method----------------------------------------------
            # ksai_output = zeros(self.output_layer_size)
            # for _ in range(self.output_layer_size):
            #     if((spike_sum_output[_] >= 1) and (_ != label)):
            #         ksai_output[_] = -1
            #     if((spike_sum_output[_] == 0) and (_ == label)):
            #         ksai_output[_] = 1
            # update weights from hidden to output
            # delta_weight_ho = dot(spike_sum_hidden.reshape((self.hidden_layer_size,1)),
            #                       ksai_output.reshape((1,self.output_layer_size))) * self.learn_rate
            # conn_ho_w = self.net.get_states()["conn_ho"]["w"]

            # update weights from input to hidden
            # derivatives_h = spike_sum_hidden.copy()
            # derivatives_h[derivatives_h>0] = 1
            # ksai_hidden = dot(array(conn_ho_w).reshape((self.hidden_layer_size,self.output_layer_size)),
            #                   ksai_output) * derivatives_h
            # delta_weight_ih = dot(spike_sum_input.reshape((self.input_layer_size,1)),
            #                       ksai_hidden.reshape((1,self.hidden_layer_size))) * self.learn_rate

            # conn_ho_w = conn_ho_w + delta_weight_ho.flatten()
            # self.net.set_states({"conn_ho": {"w": conn_ho_w}})

            # conn_ih_w = self.net.get_states()["conn_ih"]["w"]
            # conn_ih_w = conn_ih_w + delta_weight_ih.flatten()
            # self.net.set_states({"conn_ih": {"w": conn_ih_w}})
            # end_time = time.time()
            # print("Time Cost:%f" % (end_time - start_time))
        self.net.remove(spikemon_list)
        return g

    def test(self,imgs,labels,out_dir):
        spikemon_output = SpikeMonitor(self.net["output"], name='output_spikes')
        spikemon_hidden = SpikeMonitor(self.net["hidden"], name='hidden_spikes')
        spikemon_input = SpikeMonitor(self.net["input"], name='input_spikes')
        spikemon_list = [spikemon_input,spikemon_hidden,spikemon_output]
        self.net.add(spikemon_list)

        num = len(labels)
        correct_num = 0
        last_output_count = np.zeros(self.output_layer_size)
        last_hidden_count = np.zeros(self.hidden_layer_size)
        for _ in range(num):
            count_spike_sum = np.zeros(10)
            #--------------------------------FF_Method----------------------------------------------
            # for i in range(10):
            #     img_test = imgs[_].copy()
            #     img_test = overlay_label_on_img(img_test,i)
            #     self.set_input(array(img_test).flatten())
            #     self.net.run(self.sim_time * ms)
            #     current_output_count = self.net.get_states()["output_spikes"]["count"]
            #     current_hidden_count = self.net.get_states()["hidden_spikes"]["count"]
            #     increased_output_count = np.subtract(current_output_count,
            #                                     last_output_count)
            #     increased_hidden_count = np.subtract(current_hidden_count,
            #                             last_hidden_count)
            #     last_output_count = current_output_count
            #     last_hidden_count = current_hidden_count
            #     count_spike_sum[i] = self.goodness(increased_hidden_count).sum() \
            #                     #    + self.goodness(increased_output_count).sum()
            # correct_num += int(labels[_] == np.argmax(count_spike_sum))
            #--------------------------------LS_Method----------------------------------------------
            img_test = imgs[_].copy()
            img_test = overlay_label_on_img(img_test,np.random.choice(np.arange(10)))
            self.set_input(array(img_test).flatten())
            self.net.run(self.time_step * ms)
            current_output_count = self.net.get_states()["output_spikes"]["count"]
            increased_output_count = np.subtract(current_output_count,
                                            last_output_count)
            last_output_count = current_output_count
            correct_num += int(labels[_] == np.argmax(increased_output_count))
        # device.build(directory='output', compile=True, run=True, debug=False)
        accuarcy = float(correct_num) / num
        print("# Accuracy:%f" % accuarcy)
        record_file = open(f'{out_dir}/test_record_reim.txt', 'a+')
        localtime = time.asctime(time.localtime(time.time()))
        record_file.writelines('%s , Accuarcy : %f\n' % (localtime,accuarcy) )
        record_file.close()
        self.net.remove(spikemon_list)
        print(self.net)
        return accuarcy

    def save_weight(self,file_name):
        print("# Saving weights")
        f = h5py.File(file_name,'w')
        f["weight_1"] = self.net.get_states()["conn_ih"]["w"]
        f["weight_2"] = self.net.get_states()["conn_ho"]["w"]
        f.close()

    def load_weight(self,file_name):
        print("# Loading weights from %s" % file_name)
        f = h5py.File(file_name, 'r')
        self.net.set_states({"conn_ih":{"w":f["weight_1"][:]}})
        self.net.set_states({"conn_ho":{"w":f["weight_2"][:]}})
        f.close()


if __name__ == "__main__":
    f = h5py.File("./HDF5_MNIST_TRAIN_GROUPED.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()

    f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()
    out_dir = os.path.join("./log_data",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    snn = SNN()
    save_script_as(f"{out_dir}/source.py")
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(snn.__dict__))
    # if(os.path.exists(f"{out_dir}/WEIGHTS_TRAIN_FULL.h5")):
    #     snn.load_weight(f"{out_dir}/WEIGHTS_TRAIN_FULL.h5")
    max_acc = 0
    for index in range(train_pics):
        img_neg = img[index].copy()
        img_pos = overlay_label_on_img(img[index], label[index])
        label_neg = get_label_neg(label[index])
        img_neg = overlay_label_on_img(img_neg, label_neg)
        pos_goodness = snn.train(snn.sim_time, img_pos.flatten(), label[index], True)
        neg_goodness = snn.train(snn.sim_time, img_neg.flatten(), label[index], False)
        print(f"Pos - Neg = {pos_goodness - neg_goodness}")
        # snn.save_weight(f"{out_dir}/WEIGHTS_TRAIN_REIM.h5")
        if((index+1) % test_interval == 0):
            snn.save_weight(f"{out_dir}/WEIGHTS_TRAIN_FULL.h5")
            start_index = int(np.random.rand(1)[0] * (10000-test_pics))
            acc = snn.test(test_img[start_index:start_index+test_pics],test_label[start_index:start_index+test_pics],out_dir)
            if acc >= max_acc:
                max_acc = acc
                snn.save_weight(f"{out_dir}/WEIGHTS_TRAIN_BEST.h5")
    snn.load_weight(f"{out_dir}/WEIGHTS_TRAIN_BEST.h5")
    # base_index = int(np.random.rand(1)[0] * 9000)
    base_index = 0
    start_time = time.time()
    snn.test(test_img[base_index:base_index+1000],test_label[base_index:base_index+1000],out_dir)
    end_time = time.time()
    print("Used Test Time:%f" % (end_time-start_time))
