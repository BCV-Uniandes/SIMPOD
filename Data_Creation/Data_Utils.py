import numpy as np
from PyAstronomy.pyasl import AtomicNo as an
import skimage.io as io
from skimage import img_as_ubyte
import Dans_Diffraction as dif
import json
import os
import gemmi
import torch


def extract_ID(name):
     ID = name[-11:-4]
     return ID


def get_information(path):
    xtl = dif.Crystal(path)
    space_group = xtl.Symmetry.spacegroup
    space_group_number = gemmi.find_spacegroup_by_name(space_group).number
    atomic_numbers = xtl.Atoms.type
    AN = an()
    atomic_numbers = [AN.getAtomicNo(symbol) for symbol in atomic_numbers]
    if len(atomic_numbers) < 4 or len(atomic_numbers) > 256:
        return None
    occupancy = xtl.Atoms.occupancy.tolist()
    xs = xtl.Atoms.u
    ys = xtl.Atoms.v
    zs = xtl.Atoms.w
    xs = xs.tolist()
    ys = ys.tolist()
    zs = zs.tolist()
    atoms = [atomic_numbers, occupancy, xs, ys, zs]
    atoms = np.array(atoms)
    atoms = np.transpose(atoms)
    atoms = atoms.tolist()
    a = xtl.Cell.a
    b = xtl.Cell.b
    c = xtl.Cell.c
    alpha = xtl.Cell.alpha
    beta = xtl.Cell.beta
    gamma = xtl.Cell.gamma
    f = xtl.Scatter
    f._scattering_min_twotheta = 5
    f._scattering_max_twotheta = 90
    _, iten, _ = f.powder(units="twotheta")
    if np.sum(np.isnan(iten)) != 0:
        return None
    iten = iten/np.amax(iten)
    intensities = iten.tolist()
    return space_group_number, alpha, beta, gamma, a, b, c, intensities, atoms


def append_crystal_info(json_name, id, space_group, alpha, beta, gamma, a, b, c, intensities, atoms):
    data = {
            'ID': id,
            'space_group': space_group,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'a': a,
            'b': b,
            'c': c,
            'intensities': intensities,
            'atoms': atoms
        }
    with open(os.path.join('Data_Creation','Data','Structures', json_name), 'w') as f:
        json.dump(data, f, indent=2)
    f.close()


def save_powder_image(intensities, ID):
    intensities = np.array(intensities)
    intensities[intensities < 0] = 0
    intensities2 = torch.tensor(intensities).view(1,1,-1)
    intensities2 = torch.nn.functional.interpolate(intensities2, 1250).view(1250)
    intensities2 = intensities2.numpy()
    intensities2 = (intensities2/np.amax(intensities2))
    v = np.linspace(-260, 260, 521) #Full circles
    #v = np.linspace(0, 260, 521) 1/4 of circles
    x, y = np.meshgrid(v, v)
    z = np.round(((x**2 + y**2)**0.5)*5)
    Z = np.zeros(z.shape)
    for i in range(20, 20 + len(intensities2)):
        Z[z == i] = intensities2[i-20]
    io.imsave(os.path.join('Data_Creation','Data', 'Powder_images', ID+'.png'), img_as_ubyte(Z))