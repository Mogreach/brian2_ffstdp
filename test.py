from brian_reimplement import SNN
import h5py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    f = h5py.File("./HDF5_MNIST_TRAIN_GROUPED.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()

    f = h5py.File("./HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()
    snn = SNN()
    snn.load_weight(f"log_data\\2025-02-16_23-20-18\WEIGHTS_TRAIN_BEST.h5")
    weight_data = snn.net.get_states()["conn_ih"]["w"]
    # 绘制直方图
    plt.hist(weight_data, bins=300, color='blue', alpha=0.7, edgecolor='black')

    # 添加标题和标签
    plt.title("数值大小分布直方图")
    plt.xlabel("数值")
    plt.ylabel("频数")
    plt.imshow(weight_data.reshape(784,784))
    # 显示图形
    plt.show()
    print(1)