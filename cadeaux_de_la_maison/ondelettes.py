import numpy as np
import pywt


def main():
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, size=256)


    wavelet = 'sym4'
    coeffs = pywt.wavedec(returns, wavelet, level=3)
    return 0

if __name__ == "__main__":
    main()