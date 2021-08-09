"""Update an image binary preprocessing to include a 
Gaussian tapering. Then update the weights so that two 
channels provide comparable contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyink as pu


def gaussian_2d(X, Y, A, x0, y0, sig_x, sig_y, pa):
    # See https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    pa_rad = np.deg2rad(pa)
    a = 0.5 * (np.cos(pa_rad) / sig_x) ** 2 + 0.5 * (np.sin(pa_rad) / sig_y) ** 2
    b = np.sin(2 * pa_rad) * (-1 / (4 * sig_x * sig_x) + 1 / (4 * sig_y * sig_y))
    c = 0.5 * (np.sin(pa_rad) / sig_x) ** 2 + 0.5 * (np.cos(pa_rad) / sig_y) ** 2
    y = A * np.exp(
        -(
            a * (X - x0) * (X - x0)
            + 2 * b * (X - x0) * (Y - y0)
            + c * (Y - y0) * (Y - y0)
        )
    )
    return y


def kernel(img_size, f=None, fwhm_frac=None):
    """Create a 2D gaussian kernel of a certain size. Can be defined
    either by the size of the FWHM relative to the image size, or 
    the fraction of 
    """
    if isinstance(img_size, int):
        img_x = img_size
        img_y = img_size
    else:
        img_y, img_x = img_size[-2:]

    assert fwhm_frac is not None or f is not None, ValueError(
        "Either fwhm_frac or f must not be None"
    )
    if fwhm_frac is not None:
        sig_x = fwhm_frac * img_x / 2.3548200450309493
        sig_y = fwhm_frac * img_y / 2.3548200450309493
    elif f is not None:
        sig_x = img_x / np.sqrt(8 * np.log(1 / f))
        sig_y = img_y / np.sqrt(8 * np.log(1 / f))

    Y, X = np.mgrid[0:img_y, 0:img_x]
    return gaussian_2d(X, Y, 1, img_x / 2, img_y / 2, sig_x, sig_y, 0)


def taper_img(img, **kwargs):
    return img * kernel(img.shape, **kwargs)


def reweight(data, old_weights, new_weights):
    data = np.array(data)
    for chan in range(data.shape[1]):
        data[:, chan] *= new_weights[chan] / old_weights[chan]
    return data


imgs = pu.ImageReader(
    "EMU_WISE_E95E05_Aegean_Components_Complex_EMUWISE_IslandNorm_Log_Reprojected.bin"
)


# data = imgs.reweight(old_weights=(0.95, 0.05), new_weights=(1, 1))
data = reweight(imgs.data, (0.95, 0.05), (1, 1))
data = taper_img(data, fwhm_frac=0.5)

i = np.random.randint(imgs.data.shape[0])
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(imgs.data[i, 0])
axes[0, 1].imshow(imgs.data[i, 1])
axes[1, 0].imshow(data[i, 0])
axes[1, 1].imshow(data[i, 1])

rad_tot = data[:, 0].reshape(-1, 150 * 150).sum(axis=1)
ir_tot = data[:, 1].reshape(-1, 150 * 150).sum(axis=1)

rad_hist, rad_edges = np.histogram(rad_tot, bins=100, density=True)
ir_hist, ir_edges = np.histogram(ir_tot, bins=50, density=True)
plt.plot(rad_edges[:-1], rad_hist, label="EMU")
plt.plot(ir_edges[:-1], ir_hist, label="WISE")
plt.legend()

ir_weight = np.median(rad_tot) / (np.median(rad_tot) + np.median(ir_tot))
rad_weight = np.median(ir_tot) / (np.median(rad_tot) + np.median(ir_tot))
print(rad_weight, ir_weight)
data = reweight(data, (1, 1), (0.6, 0.4))
# fwhm_frac=0.5: 0.5767149112958551 0.42328508870414494

# Now create the ImageWriter
outfile = "EMU_WISE_E60E40_tapered.bin"
with pu.ImageWriter(outfile, 0, imgs.data.shape[1:], clobber=True) as pk_img:
    for i in range(imgs.data.shape[0]):
        pk_img.add(data[i], attributes=i)


# What *should* the weights have been?
rad_tot = data[:, 0].reshape(-1, np.product(imgs.data.shape[-2:])).sum(axis=1) / 0.95
ir_tot = data[:, 1].reshape(-1, np.product(imgs.data.shape[-2:])).sum(axis=1) / 0.05
ir_weight = np.median(rad_tot) / (np.median(rad_tot) + np.median(ir_tot))
rad_weight = np.median(ir_tot) / (np.median(rad_tot) + np.median(ir_tot))
print(rad_weight, ir_weight)
# 0.9655545469908711 0.03444545300912895
