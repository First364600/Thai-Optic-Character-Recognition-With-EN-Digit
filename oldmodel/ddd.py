import os
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# ตั้งค่า environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# โหลดข้อมูล MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# โหลดโมเดล
model1 = keras.models.load_model('mark.keras')

# เลือกรูปภาพที่ต้องการทำนาย
image = 6000
x = x_test[image]

# เพิ่มมิติใหม่ให้กับรูปภาพ เพื่อให้มีรูปแบบที่ถูกต้องสำหรับโมเดล
x = np.expand_dims(x, axis=0)

# ทำการ warm-up โมเดลเพื่อให้การทำนายเร็วขึ้นในครั้งต่อไป
model1.predict(x)

# วัดเวลาการทำนาย
start_time = time.time()
predict = model1.predict(x)
single_prediction_time = time.time() - start_time
print(f"Time for single prediction: {single_prediction_time:.6f} seconds")

# แสดงรูปภาพ
plt.imshow(x_test[image], cmap='binary')

# แสดงผลการทำนาย
print(np.argmax(predict[0]))

# แสดงผลภาพ
plt.show()

# Clear the session to free memory
K.clear_session()
