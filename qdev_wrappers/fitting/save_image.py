from typing import Union, Tuple, List, Optional
import os
import matplotlib
from qcodes import config

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
AxesTupleListWithRunId = Tuple[int, List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
number = Union[float, int]


def save_image(axes, exp_name=None, sample_name=None,
               run_id=None, name_extension=None, **kwargs) -> AxesTupleList:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved
            as plot.

    """
    run_id = run_id or 0
    mainfolder = config.user.mainfolder

    if exp_name and sample_name:
        storage_dir = os.path.join(mainfolder, exp_name, sample_name)
        os.makedirs(storage_dir, exist_ok=True)
        png_dir = os.path.join(storage_dir, 'png')
        pdf_dir = os.path.join(storage_dir, 'pdf')
    else:
        png_dir = 'png'
        pdf_dir = 'pdf'

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    save_pdf = config.user.get('save_pdf', True)
    save_png = config.user.get('save_png', True)

    for i, ax in enumerate(axes):
        filename = f'{run_id}'
        if name_extension is not None:
            filename += '_' + name_extension
        filename += f'_{i}'
        if save_pdf:
            full_path = os.path.join(pdf_dir, filename + '.pdf')
            ax.figure.savefig(full_path, dpi=500)
        if save_png:
            full_path = os.path.join(png_dir, filename + '.png')
            ax.figure.savefig(full_path, dpi=500)
