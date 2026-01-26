import numpy as np
file_path = r"E:\googleDownload\AthletePose3D_data_set\data\train_set\S3\Running_0_cam_1.npy"
data = np.load(file_path, allow_pickle=True)

print("数据类型：", type(data))
print("数据形状：", data.shape if hasattr(data, 'shape') else "无shape属性")
print("数据内容预览", data)