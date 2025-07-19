import os
import os.path as osp
import pandas as pd
import torch

from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable, List, Optional


class Images(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['ratings.csv', 'items.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def has_process(self) -> bool:
        return not os.path.exists(self.processed_paths[0])

    def download(self) -> None:
        pass

    def process():
        pass


class RawImages(Images, PreprocessingMixin):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None
    ) -> None:
        super(RawImages, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(self.raw_paths[0])

    def process(self, max_seq_len=None) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process item data:
        df = pd.read_csv(self.raw_paths[1], index_col='itemId')

        item_mapping = {idx: i for i, idx in enumerate(df.index)}

        # Process genres/categories as one-hot
        if 'genres' in df.columns:
            genres = self._process_genres(df["genres"].str.get_dummies('|').values, one_hot=True)
            genres = torch.from_numpy(genres).to(torch.float)
        else:
            # Create dummy genres if not available
            genres = torch.zeros((len(df), 1))

        # Process titles/descriptions as text embeddings
        if 'title' in df.columns:
            titles_text = df["title"].astype(str).tolist()
            titles_emb = self._encode_text_feature(titles_text)
        elif 'description' in df.columns:
            titles_text = df["description"].astype(str).tolist()
            titles_emb = self._encode_text_feature(titles_text)
        else:
            # Create dummy embeddings if no text features
            titles_emb = torch.zeros((len(df), 768))

        x = torch.cat([titles_emb, genres], axis=1)

        data['item'].x = x
        # Add is_train field - split items for train/eval
        num_items = len(df)
        train_ratio = 0.8
        num_train = int(num_items * train_ratio)
        is_train = torch.zeros(num_items, dtype=torch.bool)
        is_train[:num_train] = True
        data['item'].is_train = is_train
        # Add text field for consistency with other datasets
        if 'title' in df.columns:
            data['item'].text = df['title'].astype(str).tolist()
        elif 'description' in df.columns:
            data['item'].text = df['description'].astype(str).tolist()
        else:
            data['item'].text = [f"item_{i}" for i in range(len(df))]

        # Process user data:
        full_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
        df_users = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df_users["userId"])}
        self.int_user_data = df_users

        # Process rating data:
        df_ratings = self._remove_low_occurrence(
            ratings_df,
            ratings_df,
            ["userId", "itemId"]
        )
        src = [user_mapping[idx] for idx in df_ratings['userId']]
        dst = [item_mapping[idx] for idx in df_ratings['itemId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'item'].edge_index = edge_index

        # Handle rating values - convert to integers if they're floats
        if 'rating' in df_ratings.columns:
            rating_values = df_ratings['rating'].values
            if rating_values.dtype == float:
                rating = torch.from_numpy((rating_values * 2).astype(int)).to(torch.long)
            else:
                rating = torch.from_numpy(rating_values).to(torch.long)
        else:
            # Use implicit feedback (all 1s) if no explicit ratings
            rating = torch.ones(len(df_ratings), dtype=torch.long)

        data['user', 'rates', 'item'].rating = rating

        # Handle timestamp
        if 'timestamp' in df_ratings.columns:
            time = torch.from_numpy(df_ratings['timestamp'].values)
        else:
            # Use sequential timestamps if none provided
            time = torch.arange(len(df_ratings), dtype=torch.long)

        data['user', 'rates', 'item'].time = time

        # Create reverse edges
        data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'rated_by', 'user'].rating = rating
        data['item', 'rated_by', 'user'].time = time

        # Map item IDs for history generation - follow ml32m pattern exactly
        df_ratings["itemId"] = df_ratings["itemId"].apply(lambda x: item_mapping[x])
        # Only multiply rating by 2 if it exists and is a float
        if 'rating' in df_ratings.columns and df_ratings['rating'].dtype == float:
            df_ratings["rating"] = (2*df_ratings["rating"]).astype(int)
        elif 'rating' not in df_ratings.columns:
            df_ratings["rating"] = 1  # Default rating for implicit feedback

        # Generate user history
        data["user", "rated", "item"].history = self._generate_user_history(
            df_ratings,
            features=["itemId", "rating"],
            window_size=max_seq_len if max_seq_len is not None else 20,  # Smaller window for small dataset
            stride=10,  # Smaller stride for small dataset
            train_split=0.8
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
