import pandas as pd
import polars as pl
from .sessionBase import sessionBase
from ..utils.getRepositoryPath import getRepositoryPath
from glob import glob

class G1NEWS_SESSION(sessionBase):

    def read(self):
        
        paths = sorted(glob(f"{getRepositoryPath()}/dataset/G1_NEWS/clicks/clicks/clicks_hour_*.csv"))

        # Read
        df_RAW = pl.concat([
            pl.read_csv(path, dtypes={"session_id": int, "click_article_id": int, "click_timestamp": int, }).select(
                pl.lit( path.rsplit("/", 1)[1].split("_", 1)[1].replace(".csv", "") ).alias("timestamp"),
                pl.col("session_id"),
                pl.col("click_article_id").alias("item_id"),
                pl.col("click_timestamp"),
            )
            for path in paths
        ])
        df_RAW = df_RAW.sort("click_timestamp")

        # セッションに集約
        df_RAW = df_RAW.group_by(
            pl.col("timestamp"),
            pl.col("session_id"),
            maintain_order=True
        ).agg("item_id")

        # セッションIDを振ってスパース形式に展開
        df_RAW = df_RAW.select(
            pl.col("timestamp"),
            pl.arange(0, df_RAW.height).alias("session_id"),
            pl.col("item_id"),
        ).explode("item_id")

        # 連続アイテムを削除
        df_RAW = df_RAW.with_columns(
            (pl.col("session_id") == pl.col("session_id").shift()).fill_null(False).alias("session_id_remove_flag"),
            (pl.col("item_id") == pl.col("item_id").shift()).fill_null(False).alias("item_id_remove_flag"),
        ).filter(
            ~(pl.col("session_id_remove_flag") & pl.col("item_id_remove_flag"))
        )

        # 再びセッションに集約
        df_RAW = df_RAW.group_by("timestamp", "session_id").agg(pl.col("item_id").alias("item_list"))

        # 長さ1のセッションを削除
        df_RAW = df_RAW.filter(
            pl.col("item_list").list.len() > 1
        )

        # Set Variables
        self.df_RAW = df_RAW

        return self