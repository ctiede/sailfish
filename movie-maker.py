import sys
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import os
from pathlib import Path
import argparse


def file_load(indir, savefigbool):
    file_count = 0
    outdir_path_name = Path().resolve()
    Path('{}/temp'.format(outdir_path_name)).mkdir(parents=True, exist_ok=True)
    for filename in Path(indir).iterdir():
        file_count += 1
        chkpt = msgpack.load(open(filename, 'rb'))
        ni = chkpt['mesh']['ni']
        nj = chkpt['mesh']['nj']
        x0 = chkpt['mesh']['x0']
        y0 = chkpt['mesh']['y0']
        x1 = chkpt['mesh']['dx'] * chkpt['mesh']['ni'] + x0
        y1 = chkpt['mesh']['dy'] * chkpt['mesh']['nj'] + y0
        primitive = np.reshape(chkpt['primitive'], (ni + 4, nj + 4, 3))[2:-2,
                    2:-2]
        plt.figure(figsize=[12, 9.5])
        plt.imshow(primitive[:, :, 0].T ** 0.25, origin='lower', cmap='plasma',
                   extent=[x0, x1, y0, y1])
        plt.colorbar()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.title(r"{} $\Sigma^{{1/4}}$".format(filename))
        fname = '{}/temp/{}.png'.format(outdir_path_name, file_count)
        print(fname)
        plt.savefig(fname)

    if savefigbool is False:
        os.system("rm -rf {}/{}".format(outdir_path_name, 'output-figures'))


# def make_movie(outdir):
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', help='Checkpoint file directory.', required=True)
    parser.add_argument('--outdir', default='movie', help='Output movie directory.')
    parser.add_argument('--savefigs', default=False, help='Whether the program saves the figures used to make the movie.')
    args = parser.parse_args()

    file_load(args.indir, args.savefigs)
