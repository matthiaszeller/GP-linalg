

import numpy as np

def gen_tikz(A: np.ndarray,
             dx=1.0,
             cmap='binary'):
    assert A.ndim == 2

    # Colors
    assert cmap == 'binary'
    C = np.full(A.shape, 'white')
    C[A != 0] = 'black'

    header = r'\begin{tikzpicture}'
    footer = r'\end{tikzpicture}'
    items = []
    m, n = A.shape
    for i in range(m):
        ii = m-i-1
        for j in range(n):
            jj = j#n-j-1
            x1, y1 = dx*jj, dx*ii
            x2, y2 = dx*(jj+1), dx*(ii+1)
            col = C[i, j]
            items.append(
                r'\draw[fill = '+col+'] ' + f'({x1}, {y1}) rectangle ({x2}, {y2})'
            )

    snippet = header + '\n' + '\n'.join(
        f'\t{elem};' for elem in items
    ) + '\n' + footer
    return snippet


if __name__ == '__main__':
    #A = np.random.choice([0, 1], size=9).reshape(3, 3)
    #A = np.array([[1, 1, 1],[0, 0, 1],[0, 0, 1]])
    #print(gen_tikz(A))

    import torch
    from scipy.stats import ortho_group
    from src.chol import pivoted_chol

    n = 6
    Q = torch.from_numpy(ortho_group.rvs(n))
    A = Q.T @ torch.diag(torch.arange(1, n+1, dtype=torch.double)) @ Q

    L = pivoted_chol(A, k=n)

    print(gen_tikz(L))
