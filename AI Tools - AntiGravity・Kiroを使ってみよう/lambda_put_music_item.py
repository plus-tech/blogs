import json
import boto3
from botocore.exceptions import ClientError

# DynamoDB リソース（テーブル操作に便利な高レベルAPI）
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("Music")


def lambda_handler(event, context):
    """
    DynamoDB の Music テーブルにレコードを登録する Lambda 関数。

    【必須パラメータ】
      - Artist    : アーティスト名（パーティションキー）
      - SongTitle : 曲名（ソートキー）

    【任意パラメータ】（event に含まれる場合に登録）
      - AlbumTitle : アルバム名
      - Year       : リリース年（数値）
      - Genre      : ジャンル
      - Price      : 価格（数値）

    【呼び出し例（Lambda テストイベント）】
    {
        "Artist": "Acme Band",
        "SongTitle": "Happy Day",
        "AlbumTitle": "Songs About Life",
        "Year": 2000,
        "Genre": "Pop",
        "Price": 1.99
    }
    """

    # ---- 必須パラメータのバリデーション ----
    artist = event.get("Artist", "").strip()
    song_title = event.get("SongTitle", "").strip()

    if not artist:
        return _error_response(400, "必須パラメータ 'Artist' が指定されていません。")

    if not song_title:
        return _error_response(400, "必須パラメータ 'SongTitle' が指定されていません。")

    # ---- 登録アイテムの構築 ----
    item = {
        "Artist": artist,
        "SongTitle": song_title
    }

    # 任意項目が存在すれば追加
    optional_keys = ["AlbumTitle", "Year", "Genre", "Price"]
    for key in optional_keys:
        if key in event and event[key] != "" and event[key] is not None:
            item[key] = event[key]

    # ---- DynamoDB への登録 ----
    try:
        table.put_item(
            Item=item,
            # 同一キーのレコードが存在する場合は上書きせずにエラーにする場合は
            # 下記の ConditionExpression を有効化してください。
            # ConditionExpression="attribute_not_exists(Artist) AND attribute_not_exists(SongTitle)"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "レコードを正常に登録しました。",
                    "item": item
                },
                ensure_ascii=False
            )
        }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        # 条件付き書き込みで競合した場合（ConditionExpression を使用時）
        if error_code == "ConditionalCheckFailedException":
            return _error_response(
                409,
                f"Artist='{artist}', SongTitle='{song_title}' のレコードはすでに存在します。"
            )

        # テーブルが存在しない場合
        if error_code == "ResourceNotFoundException":
            return _error_response(404, "Music テーブルが見つかりません。")

        # その他の AWS エラー
        return _error_response(
            500,
            "レコードの登録中にエラーが発生しました。",
            error_code=error_code,
            error_message=e.response["Error"]["Message"]
        )

    except Exception as e:
        return _error_response(500, "予期しないエラーが発生しました。", error_message=str(e))


# ---- ヘルパー関数 ----

def _error_response(status_code: int, message: str, **kwargs) -> dict:
    """エラーレスポンスを統一フォーマットで返す。"""
    body = {"message": message}
    body.update(kwargs)
    return {
        "statusCode": status_code,
        "body": json.dumps(body, ensure_ascii=False)
    }
