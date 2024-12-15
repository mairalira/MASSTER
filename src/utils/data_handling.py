import pathlib as Path
import pandas as pd
import logging

def read_csv(data_path: Path) -> pd.DataFrame:
    """Reads a saved csv

    Parameters
    ----------
    data_path : Path
        Path of the csv file.

    Returns
    -------
    pd.DataFrame

    Raises
    -------
    RuntimeError
        If an error occurs while saving the file.
    """

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}", exc_info=True)

    return df

def save_csv(df, base_directory: str, file_name: str):
    """This is an auxiliary function to save csv files

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    base_directory : str
        string regarding to the path of the file
    file_name : str
        string with file name

    Returns
    -------
    None

    Raises
    -------
    RuntimeError
        If an error occurs while saving the file.
    """
    base_path = Path(base_directory)
    base_path.mkdir(parents=True, exist_ok=True)
    
    df_path = base_path / file_name

    try:
        df.to_csv(df_path, index=True)
    except Exception as e:
            logging.error(f"An error occurred while saving the file: {e}", exc_info=True)
    
    return None