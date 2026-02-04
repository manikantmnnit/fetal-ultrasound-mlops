
import numpy as np
import pandas as pd
from typing import List
import shutil
from config import SEED


def split_dataset(
    df,
    split_level='patient',          # 'patient' or 'plane'
    patient_col='Patient_num',
    plane_col='Plane',
    train_ratio=0.70,
    val_ratio=0.15,
    seed=None
):
    """
    Split dataset either by patient or by plane using permutation.
    """
    if seed is not None:
        np.random.seed(seed)

    # -------------------------
    # Split by PATIENT
    # -------------------------
    if split_level == 'patient':
        ids = df[patient_col].unique()

        permuted_ids = np.random.permutation(ids)
        n = len(permuted_ids)

        train_ids = permuted_ids[:int(n * train_ratio)]
        val_ids = permuted_ids[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
        test_ids = permuted_ids[int(n * (train_ratio + val_ratio)):]

        train_df = df[df[patient_col].isin(train_ids)]
        val_df   = df[df[patient_col].isin(val_ids)]
        test_df  = df[df[patient_col].isin(test_ids)]

    # -------------------------
    # Split by PLANE
    # -------------------------
    elif split_level == 'plane':
        planes = df[plane_col].unique()

        permuted_planes = np.random.permutation(planes)
        n = len(permuted_planes)

        train_planes = permuted_planes[:int(n * train_ratio)]
        val_planes   = permuted_planes[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
        test_planes  = permuted_planes[int(n * (train_ratio + val_ratio)):]

        train_df = df[df[plane_col].isin(train_planes)]
        val_df   = df[df[plane_col].isin(val_planes)]
        test_df  = df[df[plane_col].isin(test_planes)]

    else:
        raise ValueError("split_level must be 'patient' or 'plane'")

    return train_df, val_df, test_df

# merge two dataframes
def merge_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dataset_name: List[str]=["Spanish", "African"]
) -> pd.DataFrame:
    """
    Merge two datasets and add dataset identifier column.
    """
    assert len(dataset_name)==2, 'dataset_name must have exactly two entries'
    assert set(df1.columns)==set(df1.columns), 'both dataframes must have same columns'
    return pd.concat(
        [
            df1.assign(dataset=dataset_name[0]),
            df2.assign(dataset=dataset_name[1])
        ],
        ignore_index=True
    )

# To make a list of all image file names per plane
def build_plane_dict(df):
    return (
        df.groupby('Plane')['Image_name']
          .apply(list)
          .to_dict()
    )
