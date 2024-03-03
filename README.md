# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sriram V
RegisterNumber:  212222103002
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/55cb6502-a510-49eb-a176-3a8d156c4b10)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/cde3a777-705f-4ff3-bd5c-429cca21a82c)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/3bbefb8c-d959-41f4-8597-4e4f4c3b127a)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/d68a7105-3014-43b9-8536-b4b73b2e033a)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/7717f689-4beb-4fab-9203-7ee8d4e86661)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/7346bea7-a6c7-476d-889f-f3f5f15d42e6)

![image](https://github.com/Darkwebnew/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/143114486/9667103e-6744-4853-ac60-ca60bdb28945)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
