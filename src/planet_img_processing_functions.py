from pathlib import Path

from functions_library import setup_sub_dir, generate_file_list

data_dir = Path.cwd().parent.joinpath("data")

planet_imgs_path = setup_sub_dir(data_dir, "planet_images")

path_to_imgs = planet_imgs_path.joinpath("Doolow")

observation_list = generate_file_list(
        path_to_imgs, "zip", []
    )

file_names = [file_name.stem for file_name in observation_list]

