from pathlib import Path
import logging

import pandas as pd 
import numpy as np
import random

from typing import List
import shutil

# import utility functions

from utlis.data_ops import split_dataset, merge_df, build_plane_dict
from config import SEED

random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("prepare_mixed_dataset.log"),
        logging.StreamHandler()
    ]
)

current_path=Path(__file__).resolve()


# spanish dataset
spanish_df=pd.read_excel(current_path.parent.parent/"data"/"extracted"/"spanish"/"FETAL_PLANES_DB_data.xlsx")
spanish_df['Plane']=spanish_df['Plane'].apply(lambda x:x.replace(' ', '_'))
spanish_df=spanish_df[spanish_df['Plane']!='Other']
spanish_df = spanish_df[['Image_name', 'Patient_num', 'Plane']]


# african dataset
african_df=pd.read_csv(current_path.parent.parent/"data"/"extracted"/'african'/'Zenodo_dataset'/'African_planes_database.csv')

african_countries=list(african_df['Center'].unique())

# insert _ inside the plane colum
african_df['Plane']=african_df['Plane'].apply(lambda x: x.replace(' ', '_'))


african_df = african_df.rename(columns={'Filename': 'Image_name'})
african_df = african_df[['Image_name', 'Patient_num', 'Plane']]


# make dataframe
spanish_train_df, spanish_val_df, spanish_test_df = split_dataset(spanish_df)  # split by patient
spanish_train_df_plane, spanish_val_df_plane, spanish_test_df_plane = split_dataset(spanish_df, split_level = 'plane')  # split by plane


african_train_df, african_val_df, african_test_df = split_dataset(african_df)   # split by patient
african_train_df_plane, african_val_df_plane, african_test_df_plane = split_dataset(african_df, split_level = 'plane')

# split dataframes  based on patient id and planes
mixed_df_patient_split={
    'train':merge_df(df1=spanish_train_df,df2=african_train_df),
    'val':merge_df(df1=spanish_val_df,df2=african_val_df),
    'test':merge_df(df1=spanish_test_df,df2=african_test_df)
}

mixed_df_plane_split={
    'train':merge_df(df1=spanish_train_df_plane,df2=african_train_df_plane),
    'val':merge_df(df1=spanish_val_df_plane,df2=african_val_df_plane),
    'test':merge_df(df1=spanish_test_df_plane,df2=african_test_df_plane)
}


splits=['train','val','test']

# image source dir

src_dir_spanish=Path(current_path.parent.parent/'data'/'extracted'/'spanish'/'Images')
src_dir_african=Path(current_path.parent.parent/'data'/'extracted'/'african'/'Zenodo_dataset')

# image desitantion dir
dst_dir_mixed=Path(current_path.parent.parent/'data'/'mixed_images')

mixed_image_dataset_info={}
for split in splits:
    logging.info(f'Processing {split} dataset')
   
    df_split=mixed_df_patient_split[split]
    spanish_df_split = df_split[df_split['dataset'] == "Spanish"]
    african_df_split = df_split[df_split['dataset'] == "African"]

    # plane_dict
    spanish_plane_dict=build_plane_dict(spanish_df_split)
    african_plane_dict=build_plane_dict(african_df_split)
    mixed_image_dataset_info[split]={}
    for plane_sp, images_sp_list in spanish_plane_dict.items():

        # print(split, "Spanish:", plane_sp, len(images_sp_list))
        logging.info(f"{split} Spanish: {plane_sp} - {len(images_sp_list)} images")
        

        plane_dest_sp = dst_dir_mixed / split / plane_sp
        # check folder and subfolder are existing or not
        if plane_dest_sp.exists() and any(plane_dest_sp.iterdir()):
            logging.info(f"Skipping Spanish images for {split}/{plane_sp}, folder already exists")
            mixed_image_dataset_info[split][plane_sp] = {'Spanish': len(images_sp_list)}
            continue

        logging.info(f"{split} Spanish: {plane_sp} - {len(images_sp_list)} images")
        mixed_image_dataset_info[split][plane_sp] = {'Spanish': len(images_sp_list)}
        plane_dest_sp.mkdir(parents=True, exist_ok=True)

        for img_sp_name in images_sp_list:

            src_image_sp = src_dir_spanish / f"{img_sp_name}.png"
            dst_img_sp = plane_dest_sp / f"{img_sp_name}.png"

            if src_image_sp.exists():
                shutil.copy2(src_image_sp, dst_img_sp)
            else:
                logging.warning(f"⚠️ Missing Spanish: {img_sp_name}.png")
        

    # african_countries=list(african_df['Center'].unique())

    for plane_af, images_af_list in african_plane_dict.items():

        # print(split, "African:", plane_af, len(images_af_list))
        if plane_af not in mixed_image_dataset_info[split]:
            mixed_image_dataset_info[split][plane_af] = {}
        
        plane_dest_af = dst_dir_mixed / split / plane_af

        if plane_dest_af.exists() and any(plane_dest_af.iterdir()):
            logging.info(f"Skipping African images for {split}/{plane_af}, folder already exists")
            mixed_image_dataset_info[split][plane_af] = mixed_image_dataset_info[split].get(plane_af, {})
            mixed_image_dataset_info[split][plane_af]['African'] = len(images_af_list)
            continue
        logging.info(f"{split} African: {plane_af} - {len(images_af_list)} images")
        plane_dest_af.mkdir(parents=True, exist_ok=True)

        for img_af_name in images_af_list:

            # found = False

            for country in african_countries:
                src_image_af = src_dir_african / country / f"{img_af_name}.png"

                if src_image_af.exists():
                    shutil.copy2(src_image_af, plane_dest_af / f"{img_af_name}.png")
                    found = True
                    break

            if not found:
                logging.warning(f"⚠️ Missing African: {img_af_name}.png")

mixed_df_file = current_path.parent.parent / "data"/"mixed_images" / "mixed_images_dataset.csv"

# Merge patient split dataframes (or use plane split if needed)
all_mixed_df = pd.concat([mixed_df_patient_split[s].assign(split=s) for s in splits], ignore_index=True)
all_mixed_df.to_csv(mixed_df_file, index=False)
logging.info(f"Saved mixed dataframe to {mixed_df_file}")