% calculations for converting L = PC to discrete time

% plant transfer function
p = 1.0;
P_tf = tf([0.05, 1.0], [1.0, -p]) * tf([1], [0.01, 1.0])^2;

% linear controller transfer function
k_p = 4.0;
f_lp = 10.0;
beta_lp = 0.7;
f_i = 0.5;
C_tf = k_p * tf([1.0, 2.0*pi*f_i], [1.0, 0.0]) * ...
    tf([1.0], [1.0/(2*pi*f_lp)^2, (2.0*beta_lp)/(2*pi*f_lp), 1.0]);

% loop transfer function
L = P_tf * C_tf;
sys = ss(L);

sysd = ss(c2d(L, 0.001, 'zoh'));
A = sysd.A;
B = sysd.B;
C = sysd.C;
D = sysd.D;
assert( all(sysd.D == 0) );

% compute balanced realization
[sys_bal, g] = balreal(L);
sysd_bal = c2d(sys_bal, 0.001, 'zoh');
A = sysd_bal.A;
B = sysd_bal.B;
C = sysd_bal.C;
D = sysd_bal.D;

dlmwrite('vgc_A.csv', A, 'precision', 12);
dlmwrite('vgc_B.csv', B, 'precision', 12);
dlmwrite('vgc_C.csv', C, 'precision', 12);
dlmwrite('vgc_D.csv', D, 'precision', 12);

% check step response
%step(feedback(L,1));