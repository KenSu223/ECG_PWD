import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.models import load_model, Model
import time

# Loading the processed data
data_files = [f'data/Leipzing_heartbeat_DUS_FECG_{i}.npz' for i in range(1, 9)]
DUS_lists = []
ECG_lists = []

for file in data_files:
    loaded_data = np.load(file, allow_pickle=True)
    DUS_lists.append(np.array(loaded_data['DUS_list_all'], dtype=np.float32))
    ECG_lists.append(np.array(loaded_data['ECG_list_all'], dtype=np.float32))


WaveNet_1 = load_model('./WaveNet_beat/models/model_1.h5')
WaveNet_2 = load_model('./WaveNet_beat/models/model_2.h5')
WaveNet_3 = load_model('./WaveNet_beat/models/model_3.h5')
WaveNet_4 = load_model('./WaveNet_beat/models/model_4.h5')

start_time = time.time()
generated_dopplers_1 = WaveNet_1(ECG_lists[1]).numpy().reshape(np.shape(DUS_lists[1])[0],800)
generated_dopplers_2 = WaveNet_2(ECG_lists[2]).numpy().reshape(np.shape(DUS_lists[2])[0],800)
generated_dopplers_3 = WaveNet_3(ECG_lists[3]).numpy().reshape(np.shape(DUS_lists[3])[0],800)
generated_dopplers_4 = WaveNet_4(ECG_lists[4]).numpy().reshape(np.shape(DUS_lists[4])[0],800)

end_time = time.time()


execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
print(f"Len: {len(np.vstack((DUS_lists[1],DUS_lists[2],DUS_lists[3],DUS_lists[4])))}")

data_real = np.vstack((DUS_lists[1],DUS_lists[2],DUS_lists[3],DUS_lists[4]))
data_generated = np.vstack((generated_dopplers_1, generated_dopplers_2, generated_dopplers_3, generated_dopplers_4))

# Combine the datasets and create labels
X = np.vstack((data_real, data_generated))
y = np.array([0]*len(data_real) + [1]*len(data_generated))  # 0 for real, 1 for generated

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], color='blue',label='Real', alpha=0.4)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], color='red',label='Generated', alpha=0.4)
plt.legend(fontsize=13)
#plt.title('t-SNE visualization of Real vs Generated Signals')
plt.xlabel('t-SNE Component 1', fontsize=15)
plt.ylabel('t-SNE Component 2', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-72,72)
#plt.ylim(-47,47)
#plt.grid(True)
plt.savefig('./WaveNet_beat/plots/tsne_e.jpg', dpi=450)
plt.show()

'''
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1],X_tsne[y == 0, 2], color='blue',label='Real', alpha=0.4)
ax.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], X_tsne[y == 1, 2],color='red',label='Generated', alpha=0.4)
ax.legend()

ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True

ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# Rotate the plot around the x-axis by -15 degrees
ax.view_init(10,-120)

ax.set_xlabel('t-SNE Feature 1', fontsize=13)
ax.set_ylabel('t-SNE Feature 2', fontsize=13)
ax.set_zlabel('t-SNE Feature 3', fontsize=13)
#plt.grid(True)
plt.savefig('./WaveNet_beat/plots/tsne3d.jpg')
plt.show()
'''

