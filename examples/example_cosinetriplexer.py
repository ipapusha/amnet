import matplotlib.pyplot as plt
import numpy as np
import amnet

def feedforward(phi_tri,x):
	for i in phi_tri:
		return i.eval([x])

def obj_fun(phi_tri,train_x,train_y):
	Error =[]
	for x_in,y_in in zip(train_x,train_y):
		Z = feedforward(phi_tri,x_in)
		assert len(Z)==1
		Error.append(Z[0]-y_in)
	return 0.5/train_x.shape[0]*np.matmul(np.array(Error).T,np.array(Error))

def num_der(funObj,beta,x_in,y_in):
	p = len(beta)
	assert p%6 ==0
	mu = 2.*np.sqrt([1e-12])
	diff1 = np.zeros((p,))
	diff2 = np.zeros((p,))
	for j in xrange(p):
		e_j = np.zeros((p,))
		e_j[j] = 1
		diff1[j] = funObj(beta+mu*e_j,x_in,y_in)
		diff2[j] = funObj(beta-mu*e_j,x_in,y_in)
	g = (diff1-diff2)/(2.0*mu)
	g[4::6]=0
	g[5::6]=0
	return g

def plexer_function(beta,x_in,y_in):
	phi_tri = generate_plex(beta)
        return obj_fun(phi_tri,x_in,y_in)

def generate_plex(beta):
	a,b,c,d,e,f = unpack_plex(beta)
	phi_tri = []
	for i in xrange(Plexers):
		phi_tri.append( amnet.atoms.triplexer(x,a[i*4:i*4+4],b[i*4:i*4+4],c[i*4:i*4+4],d[i*4:i*4+4],e[i*4:i*4+4],f[i*4:i*4+4]))
        return phi_tri

def unpack_plex(beta):
	a = beta[0::6]
	b = beta[1::6]
	c = beta[2::6]
	d = beta[3::6]
	e = beta[4::6]
	f = beta[5::6]
	return a,b,c,d,e,f
	
def pack_plex((a,b,c,d,e,f)):
	beta =[]
	for a_i,b_i,c_i,d_i,e_i,f_i in zip(a,b,c,d,e,f):
		beta.append(a_i)
		beta.append(b_i)
		beta.append(c_i)
		beta.append(d_i)
		beta.append(e_i)
		beta.append(f_i)
	return beta

np.random.seed(1)
x = 2.0 * np.random.rand(1000)
y = np.cos(x)

epochs = 100
alpha = 1
m = 100
train_x = 2.0*np.random.rand(m)-1
train_y = np.cos(train_x)
assert train_x.shape==train_y.shape

x_test = 2.0 * np.random.rand(1000)-1
y_test = np.cos(x_test)

Plexers = 1
x = amnet.Variable(1,name='x')
a = np.zeros(Plexers*4)
a[0] = 1
a[3] = 1
b = np.zeros(Plexers*4)
b[1] = 0.9
b[2] = 0.9
c = np.zeros(Plexers*4)
c[1] =  0.709
c[2] = -0.709
c[3] = 1
d = np.zeros(Plexers*4)
d[1] = 1.249
d[2] = 1.249
e = np.zeros(Plexers*4)
e[1] =  2
e[2] = -2
e[3] =  1
f = np.zeros(Plexers*4)
f[1] = 1
f[2] = 1

beta = pack_plex((a,b,c,d,e,f))
loss_history =[]
loss_test = []

for i in xrange(epochs):
	alpha = 1./(1+i)
	f = plexer_function(beta,train_x,train_y)
	g = num_der(plexer_function,beta,train_x,train_y)
	beta -= alpha*g
	f_test = plexer_function(beta,x_test,y_test)
	loss_history.append(f)
	loss_test.append(f_test)

tri_plex = []
tri_plex.append(generate_plex(beta))
z = []
for x_i in x_test:
	z.append(feedforward(tri_plex[0],x_i))

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(loss_history,label= 'Train error')
ax[0].plot(loss_test,label='Test error')
ax[0].set_title('Error')
ax[0].legend()
ax[1].plot(x_test,y_test,label='Exact',linestyle='None', marker='x')
ax[1].plot(x_test,z,label='AMNET',linestyle='None',marker='x')
ax[1].set_title('AMNET cosine wave')
ax[1].legend()
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
plt.show()
