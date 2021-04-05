import sympy as sy
import numpy as np
### 生成Waston函数及其梯度、Hessian矩阵表达式
m = 31
n = 12  # 需要遍历2到31的整数
i, j = sy.symbols('i j')
x = sy.IndexedBase('x')

r1 = sy.Sum((j-1)*x[j-1]*(i/29)**(j-2), (j, 2, n))
r2 = sy.Sum(x[j-1]*(i/29)**(j-1), (j, 1, n))
r = r1 - r2**2 - 1
fexpr = sy.Sum(r**2, (i, 1, m-2)) + x[0]**2 + (x[1] - x[0]**2-1)**2  # 函数表达式
gexpr = [sy.diff(fexpr, x[i]) for i in range(n)] # 梯度函数表达式表
Gexpr = [[sy.diff(g, x[i]) for i in range(n)] for g in gexpr]  # Hessian矩阵表达式的表
### 生成Waston函数及其梯度、Hessian的python数值计算函数，使用numpy库
flambdify = sy.lambdify(x, fexpr, "numpy")  # 函数数值计算
glambdify = [sy.lambdify(x, gf, "numpy") for gf in gexpr]  # 梯度数值计算
Glambdify = [[sy.lambdify(x, gf, "numpy") for gf in Gs] for Gs in Gexpr]  # Hessian矩阵数值计算
### 给上述数值计算函数添加统计调用次数功能
count_f = 0 # 计数器，用来统计函数调用的次数
count_g = 0 # 计数器，用来统计梯度调用的次数
count_G = 0 # 计数器，用来统计Hessian矩阵调用的次数
def f(x):
    global count_f
    count_f += 1 # 每调用一次就将计数器的值增1
    return flambdify(x)

def g(y):
    global count_g
    count_g += 1
    return np.array([gf(y) for gf in glambdify])

def G(y):
    global count_G
    count_G += 1
    return np.array([[gf(y) for gf in Gs] for Gs in Glambdify])

### 统计CPU时间示例
import time
def find_interval(x0,dk):
    left = 0
    step = 1
    right = step + left
    k = 2
    while True:
        if f(x0 + right * dk) < f(x0 + left * dk):
            step *= k
            left = right
            right = left + step
            return left, right
        else:
            if right <= 0:
                left = right
                right = 0
            return left, right

def find_root(x0, dk, a, b):#二分法
    alpha = (a + b) / 2
    #rho = 0.25 #0-0.5
    rho = 0.01
    t = 3
    xk = x0
    gk = g(xk)
    left = a
    right = b
    while True:
        if f(xk + alpha * dk) - f(xk) <= rho * alpha * np.dot(gk, dk):#Goldstein准则
            if f(xk + alpha * dk) - f(xk) >= (1 - rho) * alpha * np.dot(gk, dk):
                return alpha
            else:
                left = alpha
                right = right
                if right >= b:
                    alpha = t * alpha
                    return alpha
                alpha = (left + right) / 2
        else:
            left = left
            right = alpha
            alpha = (left + right) / 2
        return alpha

'''
#拉格朗日插值法
def lagrange(x_val, y_val, x):
    assert len(x_val) > 1 and (len(x_val) == len(y_val))

    def basis(i):
        l_i = [(x - x_val[j]) / (x_val[i] - x_val[j]) for j in range(len(x_val)) if j != i]
        return reduce(operator.mul, l_i) * y_val[i]

    return sum(basis(i) for i in range(len(x_val)))


'''

def Damped_netwon(x0, eps): # 阻尼牛顿法  x0初始点  eps精确度
    start_time = time.time() # 开始时间
    step = 0 # 运行次数
    shape = np.shape(x0)[0]
    xk = x0 # 初始点
    gk = g(xk) # 计算梯度函数
    Gk = G(xk) # 计算Hessian函数
    sigma = np.linalg.norm(gk) # 梯度函数范数
    dk = -1 * np.dot(np.linalg.inv(Gk), gk) # 梯度方向计算
    w = np.zeros((shape, 10**3))
    while sigma > eps and step < 500:
        alpha = find_root(xk, dk, find_interval(xk, dk)[0], find_interval(xk, dk)[1])
        w[:, step] = np.transpose(xk)
        step += 1
        xk = xk + alpha * dk
        gk = g(xk)
        Gk = G(xk)
        sigma = np.linalg.norm(gk)
        dk = -1 * np.dot(np.linalg.inv(Gk), gk)
        print("==================================================================")
        print('阻尼牛顿法第{}次运行\n 运行结果：{} \n对象值：{}, 步长:{}, 运行时间：{}\n G调用{}，g调用{}，f调用{}'
                .format(step, np.array(xk), f(xk), alpha, time.time() - start_time,count_G, count_g, count_f))
    return w

def Modified_newton(x0, eps):
    start_time = time.time()
    step = 0
    shape = np.shape(x0)[0]
    eps1 = 2
    eps2 = 1
    xk = x0
    gk = g(xk)
    Gk = G(xk)
    dk = -1 * np.dot(np.linalg.inv(Gk), gk)
    sigma = np.linalg.norm(gk)
    w = np.zeros((shape, 10 ** 3))

    while sigma > eps and step < 500:
        if np.linalg.det(Gk) != 0:
            dk = dk
            if np.dot(gk.T, dk) > eps1 * np.dot(sigma, np.linalg.norm(dk)):
                dk = -1 * dk
                if abs(np.dot(gk.T, dk)) <= eps2 * np.dot(sigma, np.linalg.norm(dk)):
                    dk = -gk
        else:
            dk = -gk
        alpha = find_root(xk, dk, find_interval(xk, dk)[0], find_interval(xk, dk)[1])
        w[:, step] = np.transpose(xk)
        step += 1
        xk = xk + alpha * dk
        gk = g(xk)
        Gk = G(xk)
        sigma = np.linalg.norm(gk)
        dk = -1 * np.dot(np.linalg.inv(Gk), gk)
        print("==================================================================")
        print('修正牛顿法第{}次运行\n 运行结果：{}\n 对象值：{}, 步长：{} 运行时间：{} \n G调用{}，g调用{}，f调用{}'
                .format(step, np.array(xk), f(xk), alpha, time.time() - start_time,count_G, count_g, count_f))
        #print("G调用{}，g调用{}，f调用{}".format(count_G, count_g, count_f))
    return w

def DFP(x0, eps):
    start_time = time.time()
    step = 0
    shape = np.shape(x0)[0]
    xk = x0
    gk = g(xk)
    Gk = G(xk)
    dk = -1 * np.dot(Gk, gk)
    w = np.zeros((shape, 10 ** 8))
    while np.linalg.norm(gk) > eps and step < 1e8:
        alpha = find_root(xk, dk, find_interval(xk, dk)[0], find_interval(xk, dk)[1])
        w[:, step] = np.transpose(xk)
        step += 1
        sk = alpha * dk
        xk = xk + sk
        sk = sk.reshape(shape, 1)  # 自变量的增量
        yk = g(xk) - gk
        yk = yk.reshape(shape, 1)
        #Gk = Gk + sk.dot(sk.T)/sk.T.dot(yk) - Gk.dot(yk).dot(yk.T).dot(Gk)/yk.T.dot(Gk).dot(yk)

        Gk = Gk + sk.dot(sk.T)/sk.T.dot(yk) - Gk.dot(yk).dot(yk.T).dot(Gk)/yk.T.dot(Gk).dot(yk)
        gk = g(xk)
        Gk=np.linalg.inv(G(xk))
        dk = -1 * np.dot(Gk, gk)
        print("==================================================================")
        print("DFP法第{}次运行\n 迭代结果：{}\n 迭代值：{}，步长{}，运行时间：{}\n G调用{}，g调用{}，f调用{}"
              .format(step, np.array(xk), f(xk), alpha, time.time() - start_time,count_G, count_g, count_f))

    return w

def BFGS(x0, eps): # BFGS算法
    start_time = time.time()
    step = 0
    shape = np.shape(x0)[0]
    xk = x0
    gk = g(xk)
    Gk = G(xk)
    dk = -1 * np.dot(np.linalg.inv(Gk), gk)
    w = np.zeros((shape, 10 ** 3))
    while np.linalg.norm(gk) > eps and step < 500:
        alpha = find_root(xk, dk, find_interval(xk, dk)[0], find_interval(xk, dk)[1])
        w[:, step] = np.transpose(xk)
        step += 1
        sk = alpha * dk
        xk = xk + sk
        sk = sk.reshape(shape, 1)  # 自变量的增量
        yk = g(xk) - gk
        yk = yk.reshape(shape, 1)
        Gk = Gk + np.dot(yk, yk.T)/np.dot(yk.T, sk) - Gk.dot(sk).dot(sk.T).dot(Gk)/sk.T.dot(Gk).dot(sk)
        gk = g(xk)
        dk = -1 * np.dot(np.linalg.inv(Gk), gk)
        print("==================================================================")
        print("BFGS算法第{}次运行\n 迭代结果：{}\n迭代值：{}，步长{}，运行时间：{}\n G调用{}，g调用{}，f调用{}"
              .format(step, np.array(xk), f(xk), alpha, time.time() - start_time,count_G, count_g, count_f))

    return w

def SR1(x0, eps):
    start_time = time.time()
    step = 0
    shape = np.shape(x0)[0]
    xk = x0
    gk = g(xk)
    Gk = G(xk)
    dk = -1 * np.dot(Gk, gk)
    w = np.zeros((shape, 10 ** 3))
    while np.linalg.norm(gk) > eps and step < 500:
        alpha = find_root(xk, dk, find_interval(xk, dk)[0], find_interval(xk, dk)[1])
        w[:, step] = np.transpose(xk)
        step += 1
        sk = alpha * dk
        xk = xk + sk
        sk = sk.reshape(shape, 1) # 自变量的增量
        yk = g(xk) - gk
        yk = yk.reshape(shape, 1)
        Gk = Gk + ((sk - np.dot(Gk, yk)).dot((sk - np.dot(Gk, yk)).T) / ((sk - np.dot(Gk, yk)).T).dot(yk))
        gk = g(xk)
        dk = -1 * np.dot(Gk, gk)
        print("==================================================================")
        print("SR1法第{}次运行\n 迭代结果：{}\n 迭代值：{}，步长{}，运行时间：{}\nG调用{}，g调用{}，f调用{}"
              .format(step, np.array(xk), f(xk), alpha, time.time() - start_time,count_G, count_g, count_f))

    return w

#x0 = np.array([0,0,0,0,0,0,0,0])
#x0 = np.array([0,0,0,0,0,0])
x0 = np.zeros(n)
print(x0)

#Damped_netwon(x0, 1e-5) # 阻尼牛顿法
#Modified_newton(x0, 1e-5) # 修正牛顿法
DFP(x0, 1e-5) # DFP法
#BFGS(x0, 1e-5) # BFPS法
#SR1(x0, 1e-5) # SR1法