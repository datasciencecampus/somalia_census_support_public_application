""" Script of functions for download_data_from_ingress """

from pathlib import Path


def rm_tree(pth):

    """
    Removes everything in the chosen folder then that folder is also deleted.

    Parameters
    ----------
    pth: Path
        local data path

    Returns
    -------
    Removal of folder and it's contents from data directory
    """

    pth = Path(pth)
    if pth.exists():
        for child in pth.glob("*"):

            if child.is_file():
                child.unlink()

            else:
                rm_tree(child)

        pth.rmdir()
