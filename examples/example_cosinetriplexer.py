import numpy as np
import amnet

def feedforward(phi_tri,x):
	for i in phi_tri:
		return i.eval(x)

def obj_fun(phi_tri,train_x,train_y):
	Error =[]
	for x_in,y_in in zip(train_x,train_y):
		Z = feedForward(phi_tri,x_in)
		Error.append(Z-y_in)
	return 0.5/train_x.shape[0]*sum(Error**2)


np.random.seed(1)

epochs = 100
alpha = 1
m = 100
train_x = 2.0*np.random.rand(m)-1
train_y = np.cos(train_x)


Plexers = 1
x = amnet.Variable(1,name='x')
a = np.zeros(Plexers*4)
b = np.zeros(Plexers*4)
c = np.zeros(Plexers*4)
d = np.zeros(Plexers*4)
e = np.zeros(Plexers*4)
f = np.zeros(Plexers*4)
phi_tri = []
for i in xrange(Plexers):
	phi_tri[i] = amnet.atoms.triplexer(x,a[i*4:i*4+3],b[i*4:i*4+3],c[i*4:i*4+3],d[i*4:i*4+3],e[i*4:i*4+3],f[i*4:i*4+3])

fout = obj_fun(phi_tri=phi_tri,train_x=train_x,train_y=train_y)
print fout
