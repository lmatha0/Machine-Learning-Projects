X = []
Y = []

with open("linear_reg_data.csv", "r", encoding="utf-8-sig") as file:
    for i in file:
        line = i.strip()
        if line:
            number = i.split(",")
            X.append(float(number[0]))
            Y.append(float(number[1]))

X_mean = sum(X) / len(X)
Y_mean = sum(Y) / len(Y)

diff_X = [(i - X_mean)**2 for i in X]
diff_Y = [(i - Y_mean)**2 for i in Y]

X_std = (sum(diff_X) / len(X))**0.5
Y_std = (sum(diff_Y) / len(Y))**0.5

X_norm = [(x - X_mean) / X_std for x in X]
Y_norm = [(y - Y_mean) / Y_std for y in Y]
 
def cost(w, b):
    m = len(Y_norm)
    err = 0
    for i in range(m):
        err += ((w * X_norm[i] + b) - Y_norm[i])**2
    total_err = err / (2 * m)
    return total_err

def grad(w, b, alpha):
    m = len(Y_norm)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        y_hat = w * X_norm[i] + b
        dj_dw += (y_hat - Y_norm[i]) * X_norm[i]
        dj_db += y_hat - Y_norm[i]

    dj_dw /= m
    dj_db /= m

    w = w - alpha * dj_dw
    b = b - alpha * dj_db 

    return w, b

w, b = 0, 0
alpha = 0.01
epochs = 5000

for epoch in range(epochs):
    w, b = grad(w, b, alpha)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: cost = {cost(w,b)}")

w_unscaled = w * (Y_std / X_std)
b_unscaled = b * Y_std + Y_mean - w_unscaled * X_mean

print(f"\nTrained w: {w_unscaled}")
print(f"Trained b: {b_unscaled}")

        