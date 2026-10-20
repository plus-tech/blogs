import json
import boto3
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    """
    DynamoDB に Music テーブルを作成する Lambda 関数。
    パーティションキー: Artist (String)
    ソートキー      : SongTitle (String)
    """

    dynamodb = boto3.client("dynamodb")

    table_name = "Music"

    try:
        response = dynamodb.create_table(
            TableName=table_name,
            # キースキーマの定義
            KeySchema=[
                {
                    "AttributeName": "Artist",
                    "KeyType": "HASH"       # パーティションキー
                },
                {
                    "AttributeName": "SongTitle",
                    "KeyType": "RANGE"      # ソートキー
                }
            ],
            # 属性の型定義（キーに使う属性のみ記載）
            AttributeDefinitions=[
                {
                    "AttributeName": "Artist",
                    "AttributeType": "S"    # S = String
                },
                {
                    "AttributeName": "SongTitle",
                    "AttributeType": "S"    # S = String
                }
            ],
            # 課金モード: PAY_PER_REQUEST（オンデマンド）
            BillingMode="PAY_PER_REQUEST"
        )

        # テーブルがACTIVEになるまで待機
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(
            TableName=table_name,
            WaiterConfig={
                "Delay": 5,         # 5秒ごとにポーリング
                "MaxAttempts": 20   # 最大100秒待機
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"テーブル '{table_name}' を正常に作成しました。",
                    "tableArn": response["TableDescription"]["TableArn"]
                },
                ensure_ascii=False
            )
        }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        # テーブルがすでに存在する場合
        if error_code == "ResourceInUseException":
            return {
                "statusCode": 409,
                "body": json.dumps(
                    {"message": f"テーブル '{table_name}' はすでに存在します。"},
                    ensure_ascii=False
                )
            }

        # その他のAWSエラー
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "テーブルの作成中にエラーが発生しました。",
                    "errorCode": error_code,
                    "errorMessage": e.response["Error"]["Message"]
                },
                ensure_ascii=False
            )
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "予期しないエラーが発生しました。",
                    "errorMessage": str(e)
                },
                ensure_ascii=False
            )
        }
