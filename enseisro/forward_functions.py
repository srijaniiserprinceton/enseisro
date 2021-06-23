import jax.numpy as np
import numpy as onp

def compute_Tsr(self, s_arr):
        Tsr = np.zeros((len(s_arr), len(self.sup.gvar.r)))
        if self.sup.gvar.args.use_precomputed:
            enn1 = self.sup.nl_neighbors[self.ix, 0]
            ell1 = self.sup.nl_neighbors[self.ix, 1]
            enn2 = self.sup.nl_neighbors[self.iy, 0]
            ell2 = self.sup.nl_neighbors[self.iy, 1]
            arg_str1 = f"{enn1}.{ell1}"
            arg_str2 = f"{enn2}.{ell2}"
            U1 = self.sup.eigU[arg_str1]
            U2 = self.sup.eigU[arg_str2]
            V1 = self.sup.eigV[arg_str1]
            V2 = self.sup.eigV[arg_str2]
        else:
            m1idx = self.sup.nl_neighbors_idx[self.ix]
            m2idx = self.sup.nl_neighbors_idx[self.iy]
            U1, V1 = self.get_eig(m1idx)
            U2, V2 = self.get_eig(m2idx)
        L1sq = self.ell1*(self.ell1+1)
        L2sq = self.ell2*(self.ell2+1)
        Om1 = Omega(self.ell1, 0)
        Om2 = Omega(self.ell2, 0)
        for i in range(len(s_arr)):
            s = s_arr[i]
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            wigval = w3j(self.ell1, s, self.ell2, -1, 0, 1)
            Tsr[i, :] = -(1 - minus1pow(self.ell1 + self.ell2 + s)) * \
                Om1 * Om2 * wigval * eigfac / self.sup.gvar.r
            LOGGER.debug(" -- s = {}, eigmax = {}, wigval = {}, Tsrmax = {}"\
                         .format(s, abs(eigfac).max(), wigval, abs(Tsr[i, :]).max()))
        return Tsr
