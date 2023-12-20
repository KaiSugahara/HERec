import polars as pl
import bson
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class Pinterest(implicitBase):

    def read(self):

        # # READ
        # df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/DIGINETICA/train-item-views.csv", separator=";", null_values="NA", new_columns=["session_id", "user_id", "item_id", "timeframe", "eventdate"])

        # # READ: user_id -> pin_id
        # file_path = f'{getRepositoryPath()}/dataset/Pinterest/pinterest_iccv/subset_iccv_board_pins.bson'
        # with open(file_path, 'rb') as file:
        #     data = bson.decode_all(file.read())
        # df_USER2PIN = pl.DataFrame(data)
        # df_USER2PIN = df_USER2PIN.select( pl.col("board_id").alias("user_id"), pl.col("pins").alias("pin_id") ).explode("pin_id")

        # # READ: pin_id -> item_id
        # file_path = f'{getRepositoryPath()}/dataset/Pinterest/pinterest_iccv/subset_iccv_pin_im.bson'
        # with open(file_path, 'rb') as file:
        #     data = bson.decode_all(file.read())
        # df_PIN2ITEM = pl.DataFrame(data)

        # # CONCAT
        # df_RAW = df_USER2PIN.join( df_PIN2ITEM.select("pin_id", pl.col("im_name").alias("item_id")), how="left", on="pin_id" )

        # # Be Unique
        # df_RAW = df_RAW.unique(["user_id", "item_id"], maintain_order=True).drop("pin_id")
        
        # READ
        df_RAW = pl.concat([
            pl.read_csv("https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/pinterest-20.train.rating", separator="\t", has_header=False),
            pl.read_csv("https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/pinterest-20.test.rating", separator="\t", has_header=False),
        ])
        df_RAW = df_RAW.select( pl.col("column_1").alias("user_id"), pl.col("column_2").alias("item_id"), )

        # Set Variables
        self.df_RAW = df_RAW

        return self