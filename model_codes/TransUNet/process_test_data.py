import h5py
import os
import numpy as np

txt_file = ""
for file in os.scandir("../data/Synapse/test_vol_h5"):
    try:
        hf = h5py.File(file.path, 'r')
        print(file.name)
        name = file.name.split('.')[0]
        data_image = np.array(hf["image"][:]) 
        data_label = np.array(hf["label"][:]) 
        for i in range(len(data_image)):
            data = {"image": data_image[i], "label": data_label[i]}
            np.savez(f"../data/Synapse/test_vol_h5/{name}_{i}.npz", **data)
            txt_file += f"{name}_{i}\n"
    except:
        continue
        
f = open("./lists/lists_Synapse/test_vol.txt", "w+")
f.write(txt_file)
f.close()