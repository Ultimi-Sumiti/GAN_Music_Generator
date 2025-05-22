To compute the derivatives of  g(q)  and its norm, we first recall the vector  g(q) :
 g(q) = \begin{bmatrix} -(m_{l1} + m_{l2})g  \\ m_{l2} g  d \sin(q_2) \end{bmatrix} 

The matrix  \frac{\partial g}{\partial q}  is the Jacobian matrix of  g(q)  with respect to  q = \begin{bmatrix} q_1 \\ q_2 \end{bmatrix} :
 \frac{\partial g}{\partial q} = \begin{bmatrix} \frac{\partial g_1}{\partial q_1} & \frac{\partial g_1}{\partial q_2} \\ \frac{\partial g_2}{\partial q_1} & \frac{\partial g_2}{\partial q_2} \end{bmatrix} 

Let's compute each partial derivative:
 \frac{\partial g_1}{\partial q_1} = \frac{\partial}{\partial q_1} (-(m_{l1} + m_{l2})g ) = 0 
 \frac{\partial g_1}{\partial q_2} = \frac{\partial}{\partial q_2} (-(m_{l1} + m_{l2})g ) = 0 
 \frac{\partial g_2}{\partial q_1} = \frac{\partial}{\partial q_1} (m_{l2} g  d \sin(q_2)) = 0 
 \frac{\partial g_2}{\partial q_2} = \frac{\partial}{\partial q_2} (m_{l2} g  d \sin(q_2)) = m_{l2} g  d \cos(q_2) 

So, the Jacobian matrix  A = \frac{\partial g}{\partial q}  is:
 A = \begin{bmatrix} 0 & 0 \\ 0 & m_{l2} g  d \cos(q_2) \end{bmatrix} 

Next, we compute the norm  \|A\|  as defined:  \|A\| = \sqrt{\lambda_{\max}(A^T A)} .

First, calculate  A^T A :
 A^T = \begin{bmatrix} 0 & 0 \\ 0 & m_{l2} g  d \cos(q_2) \end{bmatrix} 
 A^T A = \begin{bmatrix} 0 & 0 \\ 0 & m_{l2} g  d \cos(q_2) \end{bmatrix} \begin{bmatrix} 0 & 0 \\ 0 & m_{l2} g  d \cos(q_2) \end{bmatrix} 
 A^T A = \begin{bmatrix} 0 & 0 \\ 0 & (m_{l2} g  d \cos(q_2))^2 \end{bmatrix} 

The eigenvalues of this diagonal matrix  A^T A  are the diagonal elements:
 \lambda_1 = 0 
 \lambda_2 = (m_{l2} g  d \cos(q_2))^2 

The maximum eigenvalue is  \lambda_{\max}(A^T A) = (m_{l2} g  d \cos(q_2))^2 .

Finally, compute the norm  \|A\| :
 \|A\| = \sqrt{\lambda_{\max}(A^T A)} = \sqrt{(m_{l2} g  d \cos(q_2))^2} = |m_{l2} g  d \cos(q_2)| 