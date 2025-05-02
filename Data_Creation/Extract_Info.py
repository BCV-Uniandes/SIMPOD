import os
import glob
import warnings
from tqdm import tqdm
from Data_Utils import extract_ID, get_information, append_crystal_info, save_powder_image

warnings.filterwarnings("ignore")
if os.path.exists(os.path.join('Data_Creation', 'Data')):
    pass
else:
    os.mkdir(os.path.join('Data_Creation','Data'))
    os.mkdir(os.path.join('Data_Creation','Data','Structures'))
    os.mkdir(os.path.join('Data_Creation','Data','Powder_images'))
ls = glob.glob(os.path.join('Data_Creation','Files','*.cif'), recursive = True)
for i in tqdm(range(len(ls))):
    ID = extract_ID(ls[i])
    try: 
        space_group_number, alpha, beta, gamma, a, b, c, intensities, atoms = get_information(ls[i])
        append_crystal_info(ID+'.json', ID, space_group_number, alpha, beta, gamma, a, b, c, intensities, atoms)
        save_powder_image(intensities, ID)
    except: 
         continue