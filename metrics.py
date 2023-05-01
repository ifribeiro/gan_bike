import numpy as np
from scipy.special import rel_entr
from scipy.stats import ks_2samp

# funções para KL-divergence
def probs(sample, bins):
    pesos = np.ones_like(sample)/len(sample)
    probs, b = np.histogram(sample, weights=pesos, bins=bins)
    return probs

def kl_divergence(original, samples):
    p_original = probs(original, bins=100)
    divergences = []
    for s in samples:
        p = probs(s, bins=100)
        p[p==0.0] = 1.e-15 # evita divisao por 0
        d = sum(rel_entr(p_original, p))
        divergences.append(d)
    return divergences

def cdf(data, bins):
    count, bins_count = np.histogram(data, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf, pdf, bins_count

def ks_teste(original, fakes, bins=50):
    """
    Quanto menor, mais parecidos são os datasets.
    """
    cdf_o, _, _ = cdf(original, bins=bins)
    ks_testes = []
    for f in fakes:
        cdf_f, _, _ = cdf(f, bins=bins)
        t = ks_2samp(cdf_o, cdf_f)[0]
        ks_testes.append(t)
    return ks_testes