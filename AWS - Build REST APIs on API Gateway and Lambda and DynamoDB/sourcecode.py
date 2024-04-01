"""
  DemoCreateTable
  Lambda function creates a table on DynamoDB.
"""

import json
import boto3 as bo
import botocore as bc

def lambda_handler(event, context):

    if event['headers'] is not None:
        dictparam = event['headers']
    elif event['queryStringParameters'] is not None:
        dictparam = event['queryStringParameters']
    elif event['body'] is not None:
        dictparam = json.loads(event['body'])
    else:
        return {
            'statusCode': 400, 
            'body': json.dumps('Name of the table to be created is not specified.')
        }

    try:
        tablename = dictparam['table']
        client = bo.client('dynamodb')
    
        response = client.create_table(
            AttributeDefinitions=[
                {
                    'AttributeName': 'Artist',
                    'AttributeType': 'S',
                },
                {
                    'AttributeName': 'SongTitle',
                    'AttributeType': 'S',
                },
            ],
            KeySchema=[
                {
                    'AttributeName': 'Artist',
                    'KeyType': 'HASH',
                },
                {
                    'AttributeName': 'SongTitle',
                    'KeyType': 'RANGE',
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5,
            },
            TableName= tablename,
        )
        
        code = 200
        msg = 'Table created'
    except bc.exceptions.ClientError as e:
        code = 500
        msg = str(e)
    except KeyError as e:
        code = 400
        msg = 'KeyError exception happened while using key {} to get the table name.'.format(str(e))

    return { 
        'statusCode': code, 
        'body': json.dumps(msg)
    }
    
    
    
"""
  DemoDeleteTable
  Lambda function deletes a table on DynamoDB.
"""

import json
import boto3 as bo
import botocore as bc

def lambda_handler(event, context):

    if event['headers'] is not None:
        dictparam = event['headers']
    elif event['queryStringParameters'] is not None:
        dictparam = event['queryStringParameters']
    elif event['body'] is not None:
        dictparam = json.loads(event['body'])
    else:
        return {
            'statusCode': 400, 
            'body': json.dumps('Name of the table to be deleted is not specified.')
        }

    try:
        tablename = dictparam['table']
        client = bo.client('dynamodb')
        
        response = client.delete_table(
            TableName = tablename,
        )
                
        code = 200
        msg = 'Table deleted'
    except bc.exceptions.ClientError as e:
        code = 500
        msg = str(e)
    except KeyError as e:
        code = 400
        msg = 'KeyError exception happened while using key {} to get the table name.'.format(str(e))
        
    return { 
        'statusCode': code, 
        'body': json.dumps(msg)
    }    
  
  
  
"""
  DemoHandleItem
  Lambda function carries on operations on a table item.
"""

import json
import boto3 as bo
import botocore as bc

def lambda_handler(event, context):

    if event['headers'] is not None:
        dictparam = event['headers']
    elif event['queryStringParameters'] is not None:
        dictparam = event['queryStringParameters']
    elif event['body'] is not None:
        dictparam = json.loads(event['body'])
    else:
        return {
            'statusCode': 400, 
            'body': json.dumps('Item to be processed is not specified.')
        }

    #
    # Add an item
    if event['httpMethod'] == 'PUT':
        
        try:
            tablename = dictparam['table']
            artist = dictparam['artist']
            songtitle = dictparam['songtitle']
            albumtitle = dictparam['albumtitle']
            
            client = bo.client('dynamodb')
            response = client.put_item(
                Item={
                    'Artist': {
                        'S': artist,
                    },
                    'AlbumTitle': {
                        'S': albumtitle,
                    },
                    'SongTitle': {
                        'S': songtitle,
                    },
                },
                ReturnConsumedCapacity='TOTAL',
                TableName = tablename,
            )
            
            code = 200
            msg = 'Item added'
        except bc.exceptions.ClientError as e:
            code = 500
            msg = str(e)
        except KeyError as e:
            code = 400
            msg = 'KeyError exception happened while using key {} to get the value.'.format(str(e))
    
        return {
            'statusCode': code,
            'body': json.dumps(msg)
        }
    #
    # Delete an item
    elif event['httpMethod'] == 'DELETE':
        try:
            tablename = dictparam['table']
            artist = dictparam['artist']
            songtitle = dictparam['songtitle']
            
            client = bo.client('dynamodb')
            response = client.delete_item(
                Key={
                    'Artist': {
                        'S': artist,
                    },
                    'SongTitle': {
                        'S': songtitle,
                    },
                },
                TableName = tablename,
            )
                    
            code = 200
            msg = 'Item deleted'
    
        except bc.exceptions.ClientError as e:
            code = 500
            msg = str(e)
        except KeyError as e:
            code = 400
            msg = 'KeyError exception happened while using key {} to get the value.'.format(str(e))
    
        return {
            'statusCode': code,
            'body': json.dumps(msg)
        }
    #
    # Select an item
    elif event['httpMethod'] == 'GET':
        try:
            tablename = dictparam['table']
            artist = dictparam['artist']
            songtitle = dictparam['songtitle']
            
            client = bo.client('dynamodb')
            response = client.get_item(
                Key={
                    'Artist': {
                        'S': artist,
                    },
                    'SongTitle': {
                        'S': songtitle,
                    },
                },
                TableName = tablename,
            )
                    
            code = 200
            if 'Item' in response.keys():
                msg = response['Item']
            else:
                msg = 'Item not found'
                
        except bc.exceptions.ClientError as e:
            code = 500
            msg = str(e)
        except KeyError as e:
            code = 400
            msg = 'KeyError exception happened while using key {} to get the value.'.format(str(e))
    
        return {
            'statusCode': code,
            'body': json.dumps(msg)
        }
    #
    # Update an item
    elif event['httpMethod'] == 'POST':
        try:
            tablename = dictparam['table']
            artist = dictparam['artist']
            songtitle = dictparam['songtitle']
            albumtitle = dictparam['albumtitle']
            
            client = bo.client('dynamodb')
            response = client.update_item(
                ExpressionAttributeNames={
                    '#AT': 'AlbumTitle',
                },
                ExpressionAttributeValues={
                    ':t': {
                        'S': albumtitle,
                    },
                },
                Key={
                    'Artist': {
                        'S': artist,
                    },
                    'SongTitle': {
                        'S': songtitle,
                    },
                },
                ReturnValues = 'ALL_NEW',
                TableName = tablename,
                UpdateExpression='SET #AT = :t',
            )
                    
            code = 200
            msg = 'Item updated'
    
        except bc.exceptions.ClientError as e:
            code = 500
            msg = str(e)
        except KeyError as e:
            code = 400
            msg = 'KeyError exception happened while using key {} to get the value.'.format(str(e))
    
        return {
            'statusCode': code,
            'body': json.dumps(msg)
        }
    #
    # Undefined request
    else:
        return {
            'statusCode': 400, 
            'body': json.dumps('Undefined request.')
        }
