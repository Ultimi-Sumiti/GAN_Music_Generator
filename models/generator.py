B(q)\ddot{q} + c(q,\dot{q})\dot{q} +g(q) = \taup_0 =
\begin{bmatrix}
0 \\
0 \\
0 \\
\end{bmatrix}
p_1 =
\begin{bmatrix}
0 \\
0 \\
q1 \\
\end{bmatrix}

p_{l1} =
\begin{bmatrix}
0 \\
0 \\
q1/2 \\
\end{bmatrix}
p_{l2} =
\begin{bmatrix}
dsin(q_2) \\
0 \\
q1 + dcos(q2) \\
\end{bmatrix}

R_1^0 =
\begin{bmatrix}
0 && 1 && 0 \\
0 && 0 && 1\\
1 && 0 && 0 \\
\end{bmatrix}

R_2^0 = R_1^0 R_2^1 =
\begin{bmatrix}
0 && 1 && 0 \\
0 && 0 && 1\\
1 && 0 && 0 \\
\end{bmatrix}
\begin{bmatrix}
cos(q2) && -sin(q2) && 0 \\
sin(q2) && cos(q2) && 0\\
0 && 0 && 1 \\
\end{bmatrix}

R_2^0 = 
\begin{bmatrix}
sin(q2) && cos(q2) && 0 \\
0 && 0 && 1\\
cos(q2) && -sin(q2) &&  \\
\end{bmatrix}

cos(q2) = c2, sin(q2) =s2

J_P^{l1} =(J_{P_1}^{l1}, J_{P_2}^{l1}) = (J_{P_1}^{l1}, 0) 
\\
J_{P_1}^{l1} = z_{j-1} = z_0 = \begin{pmatrix}	
0 && 0&& 1
\end{pmatrix}
\\
J_P^{l1}  = 
\begin{bmatrix}
0 && 0 \\
0 && 0 \\
1 && 0
\end{bmatrix}

J_O^{l1} =(J_{O_1}^{l1}, J_{O_2}^{l1}) = 
\begin{bmatrix}
0 && 0 \\
0 && 0 \\
0 && 0
\end{bmatrix}

J_P^{l2} =(J_{P_1}^{l2}, J_{P_2}^{l2}) = (J_{P_1}^{l2}, 0)  = 
\begin{bmatrix}
0 && dc2 \\
0 && 0 \\
1 && -ds2
\end{bmatrix}
\\
J_{P_1}^{l2} = z_{j-1} = z_0 = \begin{bmatrix}	
0 \\
0 \\
1
\end{bmatrix}
\\
J_{P_2}^{l2} = z_{j-1}X(p_{l2} - p_2)=
 \begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}
X
 \begin{bmatrix}
ds2 \\
0 \\
dc2
\end{bmatrix}
= \begin{bmatrix}	
dc2 \\ 
0 \\
-ds2
\end{bmatrix}

J_O^{l2} =(J_{O_1}^{l2}, J_{O_2}^{l2}) =
\begin{bmatrix}
0 && 0 \\
0 && 1 \\
0 && 0
\end{bmatrix}
\\
J_{O_1}^{l2} = 0\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
\\
 J_{O_2}^{l2} = z_{j-1} = z_1 = 
 \begin{bmatrix}
 0 \\
 1 \\
 0
 \end{bmatrix}