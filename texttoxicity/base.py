# This file contains constants for this project
import os


BASE_DIR = os.path.dirname(os.curdir)

data_path = {
    "yelp": os.path.join(BASE_DIR, "data/yelp_labelled.txt"),
    "amazon": os.path.join(BASE_DIR, "data/amazon_cells_labelled.txt"),
    "imdb": os.path.join(BASE_DIR, "data/imdb_labelled.txt"),
}
