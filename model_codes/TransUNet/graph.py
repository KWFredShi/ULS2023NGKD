from matplotlib import pyplot as plt
data = open('test.txt','r').readlines()

new_data = []
for d in data:
    s = d.split(' ')
    if s[1] != 'TRAIN:' and s[1] != 'VALIDATION:':
        continue
    new_data.append(d)

data = [[1-float(d.split(' ')[-1]), d.split(' ')[1]] for d in new_data]

train = []
val = []
x_axis_val = []
x_axis_train = []
x_value = 0
train_count = 0
train_sum = 0
for i in range(len(data)):
    if data[i][1] == 'VALIDATION:':
        val.append(data[i][0])
        x_axis_val.append(x_value)
    elif data[i+1][1] == 'VALIDATION:':
        train.append(train_sum/train_count)
        train_sum = 0
        train_count = 0
        x_axis_train.append(x_value)
    else:
        train_sum += data[i][0]
        train_count += 1
    x_value += 1

print(data)
plt.plot(x_axis_train, train, label="train")
plt.plot(x_axis_val, val, label="validation")
plt.legend(loc="upper left")
plt.title("TransUNet Training Dice Score")
plt.xlabel("Iteration")
plt.ylabel("Dice Score")
plt.show()