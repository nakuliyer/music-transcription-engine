"""Quick python script to draw graphs from the log files"""
import matplotlib.pyplot as plt
import pandas as pd

csv = pd.read_csv("good-std_gpu_model--train_spe=850.0--max_frames=100--optimizer=adam--loss=binary_crossentropy.log")
df = pd.DataFrame(csv)

#print(df)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
ax[0].plot(df["acc"])
ax[0].set_title("Accuracy per Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")

ax[1].plot(df["loss"])
ax[1].set_title("Loss per Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")

ax[2].plot(df["f1"])
ax[2].set_title("F1 Score per Epoch")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("F1 Score")
plt.savefig("good-std_gpu_model--train_spe=850.0--max_frames=100--optimizer=adam--loss=binary_crossentropy.png")
plt.show()
